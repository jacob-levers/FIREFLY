"""UpdatesController — QML bridge for the GitHub-releases updater.

Backs the Preferences → Updates section. Wraps the existing torch-free
:mod:`firefly.updater` (``fetch_latest_release`` / ``parse_release`` /
``is_newer``). The network check runs on a daemon thread and the result is
drained on a GUI-thread QTimer (the same safe cross-thread pattern the resource
meters use), so no Qt signals are emitted off-thread.
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Property, QObject, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

# Avoid `from firefly.sptpalm_analysis import __version__` — it pulls
# matplotlib.pyplot (module-level there), which builds the font cache on a
# frozen first run and stalls startup. Read the version string directly instead.
from firefly.ui._appversion import app_version
__version__ = app_version()

_API_URL = "https://api.github.com/repos/jacob-levers/FIREFLY/releases/latest"
_RELEASES_PAGE = "https://github.com/jacob-levers/FIREFLY/releases"


class UpdatesController(QObject):
    changed = Signal()
    checkingChanged = Signal()
    installingChanged = Signal()           # installing / installError flipped
    installProgressChanged = Signal()      # progress / status line advanced
    quitForUpdate = Signal()               # app MUST quit so the helper can swap + relaunch

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checking = False
        self._available = False
        self._latest_tag = ""
        self._body = ""
        self._url = _RELEASES_PAGE
        self._last_checked = "never"
        self._check_error = ""                     # non-empty → the check couldn't complete
        self._result = None                       # written off-thread, drained on GUI thread
        self._ticks = 0                            # drain ticks → bound the wait
        self._poll = QTimer(self)
        self._poll.setInterval(120)
        self._poll.timeout.connect(self._drain)
        self._MAX_TICKS = 100                      # ~12 s before giving up on a hung fetch

        # ── in-app download+install state (same off-thread→GUI-drain pattern) ──
        self._installing = False
        self._inst_progress = 0.0                  # 0..1, or -1 = indeterminate
        self._inst_status = ""
        self._inst_error = ""
        self._inst_state = ""                      # ""|downloading|installing|done|error (off-thread)
        self._inst_err_msg = ""                    # written off-thread
        self._inst_cancel = False
        self._inst_poll = QTimer(self)
        self._inst_poll.setInterval(100)
        self._inst_poll.timeout.connect(self._drain_install)

    # ── read-only state for QML ──────────────────────────────────────────
    @Property(str, constant=True)
    def version(self):
        return __version__

    @Property(bool, notify=checkingChanged)
    def checking(self):
        return self._checking

    @Property(bool, notify=changed)
    def updateAvailable(self):
        return self._available

    @Property(str, notify=changed)
    def latestTag(self):
        return self._latest_tag

    @Property(str, notify=changed)
    def releaseBody(self):
        return self._body

    @Property(str, notify=changed)
    def releaseUrl(self):
        return self._url

    @Property(str, notify=changed)
    def lastChecked(self):
        return self._last_checked

    @Property(str, notify=changed)
    def checkError(self):
        return self._check_error        # non-empty → "couldn't check" (not "up to date")

    # ── actions ──────────────────────────────────────────────────────────
    @Slot()
    def checkNow(self):
        if self._checking:
            return
        self._checking = True
        self.checkingChanged.emit()
        self._result = None
        self._ticks = 0
        self._poll.start()

        def _work():
            try:
                from firefly import updater
                rj = updater.fetch_latest_release(_API_URL)
                if not rj:
                    payload = {"ok": False,
                               "err": "Couldn't reach GitHub — check your network/proxy."}
                elif rj.get("_rate_limited"):
                    payload = {"ok": False,
                               "err": "GitHub rate limit reached on this network — try again later."}
                else:
                    rel = updater.parse_release(rj)
                    newer = bool(rel.get("tag")) and updater.is_newer(rel["tag"], __version__)
                    payload = {"ok": True, "rel": rel, "newer": newer}
            except Exception as exc:
                payload = {"ok": False, "err": str(exc)}
            # `self` may be mid-teardown; ignore if the attribute is already gone.
            try:    self._result = payload
            except Exception: pass

        threading.Thread(target=_work, daemon=True).start()

    def _drain(self):
        if self._result is None:
            self._ticks += 1
            if self._ticks >= self._MAX_TICKS:    # hung fetch → give up cleanly
                self._poll.stop()
                self._checking = False
                self.checkingChanged.emit()
                self._check_error = "Couldn't reach GitHub (timed out)."
                self._last_checked = "check failed"
                self.changed.emit()
            return
        self._poll.stop()
        res, self._result = self._result, None
        self._checking = False
        self.checkingChanged.emit()
        if res.get("ok"):
            self._check_error = ""
            self._last_checked = "just now"
            rel = res.get("rel") or {}
            self._available = bool(res.get("newer"))
            if rel.get("tag"):
                self._latest_tag = rel["tag"]
            if rel.get("body"):
                self._body = rel["body"]
            if rel.get("html_url"):
                self._url = rel["html_url"]
        else:
            # A failed fetch must NOT masquerade as "up to date".
            self._check_error = res.get("err") or "Couldn't check for updates."
            self._last_checked = "check failed"
        self.changed.emit()

    @Slot()
    def openReleasePage(self):
        QDesktopServices.openUrl(QUrl(self._url))

    # ── in-app download + install ─────────────────────────────────────────
    @Property(bool, notify=installingChanged)
    def installing(self):
        return self._installing

    @Property(float, notify=installProgressChanged)
    def installProgress(self):
        return self._inst_progress            # 0..1, or -1 → indeterminate

    @Property(str, notify=installProgressChanged)
    def installStatus(self):
        return self._inst_status

    @Property(str, notify=installingChanged)
    def installError(self):
        return self._inst_error

    @Slot()
    def downloadAndInstall(self):
        """Download the latest release's installer for this OS, verify it, stage
        the swap-and-relaunch helper, then ask the app to quit.  Network + disk
        work runs on a daemon thread; progress is drained on the GUI thread (the
        same safe pattern as ``checkNow``) so no Qt signal is emitted off-thread."""
        if self._installing:
            return
        from firefly import updater
        if not updater.is_frozen():
            self._inst_error = ("In-app update only works in the packaged app — "
                                "you're running from source, so use 'git pull'.")
            self.installingChanged.emit()
            return
        self._installing = True
        self._inst_error = ""
        self._inst_err_msg = ""
        self._inst_cancel = False
        self._inst_progress = -1.0            # indeterminate until the first byte
        self._inst_status = "Preparing…"
        self._inst_state = "downloading"
        self.installingChanged.emit()
        self.installProgressChanged.emit()
        self._inst_poll.start()

        def _work():
            try:
                from firefly import updater
                rj = updater.fetch_latest_release(_API_URL)
                rel = updater.parse_release(rj) if rj else {}
                asset = rel.get("asset")
                if not asset:
                    raise RuntimeError(
                        "The latest release has no installer for your platform. "
                        "Use 'Release notes' to download it manually.")

                def _prog(done, total):
                    self._inst_progress = (done / total) if total else -1.0

                def _status(msg):
                    self._inst_status = str(msg)

                def _cancel():
                    return self._inst_cancel

                self._inst_status = "Downloading…"
                path = updater.download_asset(
                    asset, progress_cb=_prog, status_cb=_status, cancel_cb=_cancel)
                self._inst_status = "Installing…"
                self._inst_state = "installing"
                updater.apply_update(path)        # stages helper + spawns it, returns
                self._inst_state = "done"          # GUI thread quits the app next tick
            except Exception as exc:
                self._inst_err_msg = str(exc)
                self._inst_state = "error"

        threading.Thread(target=_work, daemon=True).start()

    @Slot()
    def cancelInstall(self):
        self._inst_cancel = True

    def _drain_install(self):
        # progress/status advance every tick while downloading/installing
        self.installProgressChanged.emit()
        st = self._inst_state
        if st == "done":
            self._inst_poll.stop()
            self._inst_status = "Restarting…"
            self.installProgressChanged.emit()
            self.quitForUpdate.emit()          # app_qml quits → helper swaps + relaunches
        elif st == "error":
            self._inst_poll.stop()
            self._installing = False
            self._inst_progress = 0.0
            self._inst_status = ""
            self._inst_state = ""
            self._inst_error = self._inst_err_msg or "Update failed."
            self.installingChanged.emit()
            self.installProgressChanged.emit()
