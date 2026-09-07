"""PostprocController — re-apply an edited ROI to a finished run.

The analysis core has been able to do this for a long time
(``firefly_worker.run_postproc``: reload the run's localisations, apply new
polygons, re-run link → MSD → JDD → turning angles → dwell → clusters → figure →
every CSV into ``<run>_postprocN``).  Nothing ever called it.  This controller is
the missing wire, and it composes :class:`RunSession` rather than repeating the
spawn/drain/stop machinery — ``run_postproc`` takes the same
``(params, msg_queue, cancel_event)`` signature and speaks the same
LOG/PROGRESS/DONE/STOPPED/ERROR protocol as every other entry point.

The one thing this class must get right on its own is refusing to lie.  A run's
saved ``_localisations.csv`` is written AFTER its ROI was applied, so it holds
only what that ROI kept.  A new polygon can therefore only ever SHRINK the
region: draw a bigger one and the result silently covers the overlap while the
app reports the shape you drew.  :meth:`canApply` decides that up front, and
:meth:`start` refuses rather than producing a plausible wrong answer.
"""
from __future__ import annotations

import json
import os

from PySide6.QtCore import Property, QObject, Signal, Slot

from firefly.analysis.fa_enums import MsgKind
from firefly.ui.controllers.params.run_session import RunSession


def _run_stem(run_dir: str) -> str:
    """The run's file stem, from its localisations CSV."""
    extras = os.path.join(run_dir, "firefly_extras")
    try:
        hits = [f for f in os.listdir(extras)
                if f.endswith("_localisations.csv") and not f.startswith("._")]
    except OSError:
        return ""
    return hits[0][:-len("_localisations.csv")] if hits else ""


def _source_had_roi(run_dir: str, stem: str) -> bool:
    """Did the run that produced this folder apply an ROI?

    Either answer is safe to act on: a persisted polygon proves it, and so does
    the rasterised mask that older runs wrote instead.
    """
    extras = os.path.join(run_dir, "firefly_extras")
    if os.path.isfile(os.path.join(extras, f"{stem}_roi_mask.npy")):
        return True
    try:
        with open(os.path.join(extras, f"{stem}_params.json"),
                  encoding="utf-8") as fh:
            return bool((json.load(fh) or {}).get("roi_polygon"))
    except Exception:
        return False


def _locs_extent(run_dir: str, stem: str):
    """(y0, y1, x0, x1) of the run's saved localisations, or None."""
    csv = os.path.join(run_dir, "firefly_extras", f"{stem}_localisations.csv")
    if not os.path.isfile(csv):
        return None
    try:
        import pandas as pd
        d = pd.read_csv(csv, usecols=["x", "y"])
        if not len(d):
            return None
        return (float(d["y"].min()), float(d["y"].max()),
                float(d["x"].min()), float(d["x"].max()))
    except Exception:
        return None


class PostprocController(QObject):
    """Dispatch ``run_postproc`` for a run whose ROI has just been edited."""

    runningChanged = Signal()
    progressChanged = Signal()
    logLine = Signal(str)
    toast = Signal(str)
    finished = Signal(str)          # the new run folder
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = RunSession(self)
        self._running = False
        self._progress = 0
        self._status = ""
        self._out_dir = ""

    # ── state for QML ────────────────────────────────────────────────────
    @Property(bool, notify=runningChanged)
    def running(self):
        return self._running

    @Property(int, notify=progressChanged)
    def progress(self):
        return self._progress

    @Property(str, notify=progressChanged)
    def status(self):
        return self._status

    @Property(str, notify=runningChanged)
    def lastOutputDir(self):
        return self._out_dir

    # ── the guard ────────────────────────────────────────────────────────
    @Slot(str, "QVariantList", result="QVariantMap")
    def canApply(self, run_dir, polygons):
        """Whether these polygons can honestly be applied to this run.

        Returns ``{ok, reason}``.  ``ok`` False means the new region reaches
        outside what the source run's own ROI kept, so the localisations there
        are simply not in its output and no amount of re-processing will bring
        them back — only a fresh analysis from the movie can widen an ROI.
        """
        run_dir = str(run_dir or "")
        if not run_dir or not os.path.isdir(run_dir):
            return {"ok": False, "reason": "That run folder is not available."}
        polys = [p for p in (polygons or []) if p and len(p) >= 3]
        if not polys:
            return {"ok": False,
                    "reason": "Draw at least one region before applying."}
        stem = _run_stem(run_dir)
        if not stem:
            return {"ok": False,
                    "reason": ("This run has no saved localisations, so its ROI "
                               "cannot be changed after the fact. Re-run the "
                               "analysis from the movie.")}
        if not _source_had_roi(run_dir, stem):
            return {"ok": True, "reason": ""}      # full field saved — anything goes

        extent = _locs_extent(run_dir, stem)
        if extent is None:
            return {"ok": True, "reason": ""}
        y0, y1, x0, x1 = extent
        for poly in polys:
            ys = [float(pt[0]) for pt in poly]
            xs = [float(pt[1]) for pt in poly]
            if (min(ys) < y0 - 1.0 or max(ys) > y1 + 1.0
                    or min(xs) < x0 - 1.0 or max(xs) > x1 + 1.0):
                return {"ok": False, "reason": (
                    "This run already had an ROI, and the region you have drawn "
                    "reaches outside it.\n\n"
                    "A run only stores the localisations its own ROI kept, so "
                    "the ones outside are not there to re-analyse — the result "
                    "would silently cover just the overlap.\n\n"
                    "Draw a region inside the existing one, or re-run the "
                    "analysis from the original movie to widen it.")}
        return {"ok": True, "reason": ""}

    # ── dispatch ─────────────────────────────────────────────────────────
    @Slot(str, "QVariantList", result=bool)
    def start(self, run_dir, polygons):
        """Re-analyse ``run_dir`` with ``polygons`` into ``<run_dir>_postprocN``."""
        if self._running:
            return False
        verdict = self.canApply(run_dir, polygons)
        if not verdict.get("ok"):
            self.failed.emit(verdict.get("reason", "Cannot apply this ROI."))
            return False

        polys = [[[float(pt[0]), float(pt[1])] for pt in poly]
                 for poly in polygons if poly and len(poly) >= 3]
        from firefly import firefly_worker

        self._progress = 0
        self._status = "Starting…"
        self._out_dir = ""
        self.progressChanged.emit()

        ok = self._session.start(
            firefly_worker.run_postproc,
            # output_folder omitted → the worker picks <run_dir>_postprocN
            {"source_folder": str(run_dir), "new_polygons": polys},
            name="FIREFLY-PostprocWorker",
            on_log=lambda line: self.logLine.emit(str(line)),
            on_progress=self._on_progress,
            terminal={MsgKind.DONE: self._on_done,
                      MsgKind.STOPPED: self._on_stopped,
                      MsgKind.ERROR: self._on_error},
            on_done_no_payload=lambda: self._finish(
                "Finished — the result didn't reach the UI"),
            on_dead_error=self._on_error,
            on_finished=self._clear_running)
        if ok:
            self._running = True
            self.runningChanged.emit()
        return bool(ok)

    @Slot()
    def stop(self):
        self._session.stop()

    # ── worker callbacks ─────────────────────────────────────────────────
    def _on_progress(self, payload):
        try:
            pct, msg = payload
        except Exception:
            return
        self._progress = int(pct)
        self._status = str(msg)
        self.progressChanged.emit()

    def _on_done(self, payload):
        payload = payload or {}
        # run_postproc adds these two precisely so the GUI can offer to open the
        # new run — see its DONE emit.
        self._out_dir = str(payload.get("postproc_output")
                            or payload.get("out_dir") or "")
        n_tracks = payload.get("n_tracks")
        where = os.path.basename(self._out_dir.rstrip(os.sep)) or "the new run"
        self._finish(f"ROI applied — {n_tracks:,} trajectories in {where}"
                     if isinstance(n_tracks, int)
                     else f"ROI applied — wrote {where}")
        self.finished.emit(self._out_dir)

    def _on_stopped(self, _payload=None):
        self._finish("Stopped before finishing")

    def _on_error(self, payload=None):
        msg = str(payload or "Post-processing failed")
        self._status = msg
        self.progressChanged.emit()
        self.failed.emit(msg)

    def _finish(self, message):
        self._progress = 100
        self._status = message
        self.progressChanged.emit()
        self.toast.emit(message)

    def _clear_running(self):
        self._running = False
        self.runningChanged.emit()
