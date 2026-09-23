"""Pytest bootstrap — make the project modules importable from tests/ and
force UTF-8 stdout so the analysis code's non-ASCII progress prints don't
raise UnicodeEncodeError on a cp1252 Windows console (the app itself
redirects stdout, so this only matters under a bare pytest console)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ── stray Qt timers ──────────────────────────────────────────────────────────
# Controllers own QTimers that keep running after the test that built them has
# finished.  On their own they are harmless; the damage appears much later, when
# some other test spins a long ``processEvents()`` loop and a 250ms-1s timer
# finally fires into a controller whose window is already gone — a segmentation
# fault attributed to whichever test happened to be running, not the one that
# leaked.  That is exactly how the Qt CI job failed intermittently.
#
# Stopping every still-active timer after each test removes the whole class of
# failure, and is safe because no test needs a timer to outlive it.  The scan is
# skipped entirely unless Qt is already imported AND the test just used it, so
# the analysis-only suite pays nothing.
def pytest_runtest_teardown(item, nextitem):
    qtcore = sys.modules.get("PySide6.QtCore")
    if qtcore is None:
        return
    # Controller worker threads call back into their controller (queue puts,
    # signal emits).  One still running when the controller is destroyed writes
    # into freed memory, and the fault surfaces inside a LATER test's
    # processEvents loop.  Individual files had their own waits, but helpers get
    # imported across files and the waits did not come with them — so do it once,
    # here, for every test.
    try:
        import threading
        import time as _time
        jobs = {"_FigureJob", "_PanelJob", "_GroupAllPanelsJob",
                "_ReportJob", "_EngineFigJob"}
        deadline = _time.monotonic() + 20.0
        while _time.monotonic() < deadline:
            alive = [t for t in threading.enumerate()
                     if t.is_alive() and (type(t).__name__ in jobs
                                          or str(t.name).startswith("FIREFLY-"))]
            if not alive:
                break
            app = qtcore.QCoreApplication.instance()
            if app is not None:
                app.processEvents()
            _time.sleep(0.02)
    except Exception:
        pass
    try:
        import gc

        import shiboken6
        timer_cls = qtcore.QTimer
        for obj in gc.get_objects():
            if type(obj) is timer_cls:
                try:
                    if shiboken6.isValid(obj) and obj.isActive():
                        obj.stop()
                except Exception:
                    pass
    except Exception:
        pass
