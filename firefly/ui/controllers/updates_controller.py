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

from firefly.sptpalm_analysis import __version__

_API_URL = "https://api.github.com/repos/jacob-levers/FIREFLY/releases/latest"
_RELEASES_PAGE = "https://github.com/jacob-levers/FIREFLY/releases"


class UpdatesController(QObject):
    changed = Signal()
    checkingChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checking = False
        self._available = False
        self._latest_tag = ""
        self._body = ""
        self._url = _RELEASES_PAGE
        self._last_checked = "never"
        self._result = None                       # written off-thread, drained on GUI thread
        self._ticks = 0                            # drain ticks → bound the wait
        self._poll = QTimer(self)
        self._poll.setInterval(120)
        self._poll.timeout.connect(self._drain)
        self._MAX_TICKS = 100                      # ~12 s before giving up on a hung fetch

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
                rel = updater.parse_release(rj) if rj else {}
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
                self._last_checked = "check failed"
                self.changed.emit()
            return
        self._poll.stop()
        res, self._result = self._result, None
        self._checking = False
        self.checkingChanged.emit()
        self._last_checked = "just now"
        if res.get("ok"):
            rel = res.get("rel") or {}
            self._available = bool(res.get("newer"))
            if rel.get("tag"):
                self._latest_tag = rel["tag"]
            if rel.get("body"):
                self._body = rel["body"]
            if rel.get("html_url"):
                self._url = rel["html_url"]
        self.changed.emit()

    @Slot()
    def openReleasePage(self):
        QDesktopServices.openUrl(QUrl(self._url))
