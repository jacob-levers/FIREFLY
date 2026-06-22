"""CudaController — QML bridge for the in-app CUDA (GPU torch) installer.

Backs Preferences ▸ GPU acceleration.  Wraps :mod:`firefly.cuda_installer`:
detect the NVIDIA GPU, report the bundled-vs-installed sidecar torch version,
and install / update / remove the CUDA wheel that lets the analysis worker run
on the GPU.

Every cuda_installer call is potentially slow or blocking — ``detect_nvidia_gpu``
shells out to ``nvidia-smi`` (5 s timeout), ``bundled_torch_version`` imports
torch, and install downloads + extracts a multi-GB wheel — so they all run on
daemon threads.  Results are written to plain fields and drained on a GUI-thread
``QTimer`` (the same safe pattern as :class:`UpdatesController`), so no Qt signal
is ever emitted off-thread.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot


class CudaController(QObject):
    changed = Signal()             # gpu / installed / versions refreshed
    busyChanged = Signal()         # install/uninstall started or finished (+ error)
    progressChanged = Signal()     # download/extract progress or status advanced

    def __init__(self, parent=None):
        super().__init__(parent)
        self._supported = False     # platform can use a CUDA sidecar (Windows)
        self._checked = False       # at least one refresh has completed
        self._gpu = ""              # detected NVIDIA GPU name
        self._installed = False     # a CUDA torch sidecar is installed + ABI-ok
        self._installed_ver = ""    # sidecar torch version (e.g. 2.12.0+cu130)
        self._bundled_ver = ""      # bundled CPU torch version
        self._busy = False
        self._progress = -1.0       # 0..1, or -1 → indeterminate
        self._status = ""
        self._error = ""

        # off-thread → GUI-drain scratch
        self._refresh_result = None
        self._refresh_pending = False
        self._inst_state = ""       # ""|done|error (written off-thread)
        self._inst_progress = -1.0
        self._inst_status = ""
        self._inst_err = ""
        self._cancel = False
        self._poll = QTimer(self)
        self._poll.setInterval(150)
        self._poll.timeout.connect(self._drain)

    # ── read-only state for QML ───────────────────────────────────────────
    @Property(bool, notify=changed)
    def supported(self):
        return self._supported

    @Property(bool, notify=changed)
    def checked(self):
        return self._checked

    @Property(str, notify=changed)
    def gpuName(self):
        return self._gpu

    @Property(bool, notify=changed)
    def installed(self):
        return self._installed

    @Property(str, notify=changed)
    def installedVersion(self):
        return self._installed_ver

    @Property(str, notify=changed)
    def bundledVersion(self):
        return self._bundled_ver

    @Property(bool, notify=busyChanged)
    def busy(self):
        return self._busy

    @Property(float, notify=progressChanged)
    def progress(self):
        return self._progress

    @Property(str, notify=progressChanged)
    def status(self):
        return self._status

    @Property(str, notify=busyChanged)
    def error(self):
        return self._error

    # ── refresh GPU + installed state (lazy; call when the section shows) ──
    @Slot()
    def refresh(self):
        if self._refresh_pending or self._busy:
            return
        self._refresh_pending = True
        self._poll.start()

        def _work():
            res = {"supported": False, "gpu": "", "installed": False,
                   "iver": "", "bver": ""}
            try:
                from firefly import cuda_installer as ci
                res["supported"] = bool(ci.is_windows())
                res["gpu"] = ci.detect_nvidia_gpu() or ""
                res["installed"] = bool(ci.is_installed())
                res["iver"] = ci.installed_torch_version() or ""
                res["bver"] = ci.bundled_torch_version() or ""
            except Exception:
                pass
            self._refresh_result = res

        threading.Thread(target=_work, daemon=True).start()

    # ── install / update the CUDA torch sidecar ───────────────────────────
    @Slot()
    def install(self):
        if self._busy:
            return
        self._busy = True
        self._error = ""
        self._inst_err = ""
        self._cancel = False
        self._inst_state = ""
        self._inst_progress = -1.0
        self._inst_status = "Preparing…"
        self.busyChanged.emit()
        self.progressChanged.emit()
        self._poll.start()

        def _work():
            try:
                from firefly import cuda_installer as ci
                ci.set_log_callback(
                    lambda m: setattr(self, "_inst_status", str(m)[:200]))
                if not ci.is_windows():
                    raise RuntimeError(
                        "CUDA acceleration is Windows-only. This Mac already uses "
                        "the built-in Metal (MPS) GPU backend — no install needed.")
                if ci.detect_nvidia_gpu() is None:
                    raise RuntimeError(
                        "No NVIDIA GPU detected (nvidia-smi). Install the NVIDIA "
                        "driver first, then try again.")
                bundled = ci.bundled_torch_version()
                if not bundled:
                    raise RuntimeError("Couldn't read the bundled torch version.")

                def _dl(done, total):
                    self._inst_progress = (done / total) if total else -1.0

                def _ex(done, total):
                    self._inst_progress = (done / total) if total else -1.0

                def _st(msg):
                    self._inst_status = str(msg)[:200]

                def _cancel():
                    return self._cancel

                ci.install_cuda_torch_auto(
                    bundled, download_progress_cb=_dl, extract_progress_cb=_ex,
                    status_cb=_st, cancel_cb=_cancel)
                self._inst_state = "done"
            except Exception as exc:
                self._inst_err = str(exc)
                self._inst_state = "error"
            finally:
                try:
                    from firefly import cuda_installer as ci
                    ci.set_log_callback(None)
                except Exception:
                    pass

        threading.Thread(target=_work, daemon=True).start()

    @Slot()
    def cancel(self):
        self._cancel = True

    @Slot()
    def uninstall(self):
        if self._busy:
            return
        try:
            from firefly import cuda_installer as ci
            ci.uninstall()
        except Exception:
            pass
        self.refresh()

    # ── GUI-thread drain ──────────────────────────────────────────────────
    def _drain(self):
        if self._refresh_result is not None:
            r, self._refresh_result = self._refresh_result, None
            self._refresh_pending = False
            self._supported = r["supported"]
            self._gpu = r["gpu"]
            self._installed = r["installed"]
            self._installed_ver = r["iver"]
            self._bundled_ver = r["bver"]
            self._checked = True
            self.changed.emit()

        if self._busy:
            self._progress = self._inst_progress
            self._status = self._inst_status
            self.progressChanged.emit()
            st = self._inst_state
            if st == "done":
                self._busy = False
                self._inst_state = ""
                self._status = "Installed — restart FIREFLY to run on the GPU."
                self.busyChanged.emit()
                self.progressChanged.emit()
                self.refresh()
            elif st == "error":
                self._busy = False
                self._inst_state = ""
                self._error = self._inst_err or "Install failed."
                self._status = ""
                self.busyChanged.emit()
                self.progressChanged.emit()

        if not self._refresh_pending and not self._busy:
            self._poll.stop()
