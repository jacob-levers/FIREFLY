"""HyperflyController — live dashboard for the parallel-batch (HYPER-FLY) run.

When a folder batch runs on a big workstation (≥32 cores / ≥192 GB RAM) FIREFLY
fans the queue out across ``n_concurrent`` worker processes
(``firefly_worker._run_batch_hyperfly``).  The worker stream carries per-file
signals — ``HYPERFLY_STATUS`` (how many run at once), ``HF_TILE``
(``{file, stem, state, pct, stage, n_locs, n_tracks}``), and ``PREVIEW_FRAME``
(a downscaled live frame tagged with its ``file`` index).

``HF_TILE.file`` is the FILE index, not a stable worker slot (the pool decides
which process runs which file), so this controller recreates the SPEC's *fixed
worker tiles* by assigning each running file to a stable slot and freeing it when
the file finishes — files flow through ``n_concurrent`` stable slots while the
grid never reshuffles.  Fed by BatchController; nothing here touches the worker.
"""
from __future__ import annotations

import time

from PySide6.QtCore import (Property, QAbstractListModel, QByteArray, QModelIndex,
                            QObject, Qt, QTimer, Signal, Slot)
from PySide6.QtQuick import QQuickImageProvider
from PySide6.QtGui import QImage


class HfWorkerModel(QAbstractListModel):
    """One row per HYPER-FLY worker slot; reads the owner's live slot state."""

    _ROLES = ["slot", "stem", "state", "pct", "stage", "locs", "tracks",
              "frameToken", "hasFrame"]

    def __init__(self, owner):
        super().__init__(owner)
        self._owner = owner
        self._roles = {Qt.UserRole + 1 + i: QByteArray(n.encode())
                       for i, n in enumerate(self._ROLES)}
        self._names = {Qt.UserRole + 1 + i: n for i, n in enumerate(self._ROLES)}

    def roleNames(self):
        return dict(self._roles)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._owner._slots)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if not (0 <= row < len(self._owner._slots)):
            return None
        name = self._names.get(role)
        return self._owner._row_dict(row).get(name) if name else None

    def reset(self):
        self.beginResetModel()
        self.endResetModel()

    def rowChanged(self, row):
        if 0 <= row < len(self._owner._slots):
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, list(self._roles.keys()))


class HfWorkerFrameProvider(QQuickImageProvider):
    """Serves per-slot live frames — ``image://hfworker/<slot>_<token>``."""

    def __init__(self, controller):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._ctrl = controller

    def requestImage(self, image_id, size, requested):  # noqa: N802
        try:
            slot = int(str(image_id).split("_", 1)[0])
        except Exception:
            slot = -1
        img = self._ctrl._frames.get(slot)
        if img is None or img.isNull():
            img = QImage(1, 1, QImage.Format.Format_RGB888)
            img.fill(0)
        if size is not None:
            size.setWidth(img.width()); size.setHeight(img.height())
        return img


class HyperflyController(QObject):
    activeChanged = Signal()
    workersChanged = Signal()       # structural (slot count) change
    aggregateChanged = Signal()     # done / elapsed / throughput / eta

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n = 0                 # n_concurrent (tile count)
        self._active = False
        self._reason = ""
        self._slots = []            # list[dict] — one per worker slot
        self._file_slot = {}        # file index → slot index
        self._frames = {}           # slot → QImage
        self._total = 0
        self._done = 0
        self._failed = 0
        self._start_t = 0.0
        self._running = False
        self._model = HfWorkerModel(self)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.aggregateChanged)

    # ── model / provider access ───────────────────────────────────────────
    @Property(QObject, constant=True)
    def workerModel(self):
        return self._model

    def worker_frame_image(self, slot):
        return self._frames.get(int(slot))

    def _idle_slot(self):
        return {"file": None, "stem": "", "state": "idle", "pct": 0,
                "stage": "", "locs": 0, "tracks": 0, "frame_token": 0}

    def _row_dict(self, row):
        d = self._slots[row]
        return {"slot": row + 1, "stem": d["stem"], "state": d["state"],
                "pct": d["pct"], "stage": d["stage"], "locs": d["locs"],
                "tracks": d["tracks"], "frameToken": d["frame_token"],
                "hasFrame": row in self._frames}

    # ── lifecycle (called by BatchController) ─────────────────────────────
    @Slot(int)
    def start(self, total):
        """A batch run is starting — reset the dashboard.  Whether it's actually
        a HYPER-FLY (parallel) run is confirmed later by HYPERFLY_STATUS."""
        self._total = int(total)
        self._done = 0
        self._failed = 0
        self._active = False
        self._running = True
        self._n = 0
        self._slots = []
        self._file_slot = {}
        self._frames = {}
        self._start_t = time.monotonic()
        self._model.reset()
        self._timer.start()
        self.activeChanged.emit()
        self.workersChanged.emit()
        self.aggregateChanged.emit()

    @Slot()
    def finish(self):
        self._running = False
        self._timer.stop()
        # mark any still-running slot idle
        for s in self._slots:
            if s["state"] == "running":
                s["state"] = "idle"
        self._model.reset()
        self.aggregateChanged.emit()

    def onStatus(self, payload):
        payload = payload or {}
        self._active = bool(payload.get("active"))
        self._reason = str(payload.get("reason", ""))
        n = int(payload.get("n_concurrent") or 0)
        if n != self._n or len(self._slots) != n:
            self._n = n
            self._slots = [self._idle_slot() for _ in range(n)]
            self._file_slot = {}
            self._frames = {}
            self._model.reset()
        self.activeChanged.emit()
        self.workersChanged.emit()
        self.aggregateChanged.emit()

    def onTile(self, payload):
        payload = payload or {}
        f = payload.get("file")
        if f is None:
            return
        f = int(f)
        slot = self._slot_for_file(f, claim=True)
        if slot is None:
            return
        d = self._slots[slot]
        state = payload.get("state")
        if state == "running":
            d["state"] = "running"; d["pct"] = 0; d["stage"] = ""; d["locs"] = 0
        elif state == "done":
            d["state"] = "done"; d["pct"] = 100
            d["locs"] = int(payload.get("n_locs") or d["locs"])
            d["tracks"] = int(payload.get("n_tracks") or d["tracks"])
        elif state == "failed":
            d["state"] = "failed"
        if payload.get("stem"):
            d["stem"] = str(payload["stem"])
        if payload.get("pct") is not None:
            d["pct"] = int(payload["pct"])
        if payload.get("stage"):
            d["stage"] = str(payload["stage"])
        if payload.get("n_locs") is not None:
            d["locs"] = int(payload["n_locs"])
        self._model.rowChanged(slot)
        self.aggregateChanged.emit()

    def onPreview(self, payload):
        payload = payload or {}
        f = payload.get("file")
        if f is None:
            return
        f = int(f)
        slot = self._slot_for_file(f, claim=False)
        if slot is None:
            return
        img = self._render(payload)
        if img is not None and not img.isNull():
            self._frames[slot] = img
            self._slots[slot]["frame_token"] += 1
            self._model.rowChanged(slot)

    def onProgress(self, done, total, failed=0):
        self._done = int(done)
        self._failed = int(failed)
        if total:
            self._total = int(total)
        self.aggregateChanged.emit()

    # ── slot assignment (file → stable tile) ──────────────────────────────
    def _slot_for_file(self, f, *, claim):
        s = self._file_slot.get(f)
        if s is not None:
            return s
        if not claim or not self._slots:
            return None
        # claim a free slot (idle / finished); free its previous file mapping
        for i, d in enumerate(self._slots):
            if d["state"] in ("idle", "done", "failed"):
                old = d.get("file")
                if old is not None and self._file_slot.get(old) == i:
                    del self._file_slot[old]
                self._file_slot[f] = i
                self._slots[i] = self._idle_slot()
                self._slots[i]["file"] = f
                self._slots[i]["state"] = "running"
                self._frames.pop(i, None)
                return i
        return None

    def _render(self, payload):
        try:
            import numpy as np
            from firefly.ui.controllers.providers.live_frame_provider import render_frame
            shape = payload.get("shape") or [0, 0]
            blob = payload.get("frame")
            if not (blob and shape[0] and shape[1]):
                return None
            arr = np.frombuffer(blob, dtype=np.float32).reshape(shape[0], shape[1])
            return render_frame(arr, payload.get("xs", []), payload.get("ys", []))
        except Exception:
            return None

    # ── aggregate properties ──────────────────────────────────────────────
    @Property(bool, notify=activeChanged)
    def active(self):
        return self._active

    @Property(bool, notify=aggregateChanged)
    def running(self):
        return self._running

    @Property(str, notify=activeChanged)
    def reason(self):
        return self._reason

    @Property(int, notify=workersChanged)
    def workerCount(self):
        return self._n

    @Property(int, notify=aggregateChanged)
    def total(self):
        return self._total

    @Property(int, notify=aggregateChanged)
    def done(self):
        return self._done

    @Property(int, notify=aggregateChanged)
    def overallPct(self):
        return int(round(100.0 * self._done / self._total)) if self._total else 0

    @Property(int, notify=aggregateChanged)
    def runningCount(self):
        return sum(1 for s in self._slots if s["state"] == "running")

    @Property(int, notify=aggregateChanged)
    def queuedCount(self):
        return max(0, self._total - self._done - self.runningCount)

    def _elapsed_s(self):
        return max(0, int(time.monotonic() - self._start_t)) if self._start_t else 0

    @Property(str, notify=aggregateChanged)
    def elapsed(self):
        s = self._elapsed_s()
        return f"{s // 60}:{s % 60:02d}"

    @Property(str, notify=aggregateChanged)
    def throughput(self):
        s = self._elapsed_s()
        if self._done <= 0 or s <= 0:
            return "— /min"
        return f"{self._done / (s / 60.0):.1f} /min"

    @Property(str, notify=aggregateChanged)
    def eta(self):
        s = self._elapsed_s()
        rem = max(0, self._total - self._done)
        if rem == 0 and self._total:
            return "done"
        if self._done <= 0 or s <= 0:
            return "—"
        rate = self._done / (s / 60.0)
        return f"{int(round(rem / rate))} min" if rate > 0 else "—"
