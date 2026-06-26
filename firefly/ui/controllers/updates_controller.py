"""UpdatesController — QML bridge for the GitHub-releases updater.

Backs the Preferences → Updates section. Wraps the existing torch-free
:mod:`firefly.updater` (``fetch_latest_release`` / ``parse_release`` /
``is_newer``). The network check runs on a daemon thread and the result is
drained on a GUI-thread QTimer (the same safe cross-thread pattern the resource
meters use), so no Qt signals are emitted off-thread.
"""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import Property, QObject, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices

# Avoid `from firefly.sptpalm_analysis import __version__` — it pulls
# matplotlib.pyplot (module-level there), which builds the font cache on a
# frozen first run and stalls startup. Read the version string directly instead.
from firefly.ui._appversion import app_version
__version__ = app_version()

_RELEASES_API = "https://api.github.com/repos/jacob-levers/FIREFLY/releases?per_page=30"
_RELEASES_PAGE = "https://github.com/jacob-levers/FIREFLY/releases"


class UpdatesController(QObject):
    changed = Signal()
    checkingChanged = Signal()
    installingChanged = Signal()           # installing / installError flipped
    installProgressChanged = Signal()      # progress / status line advanced
    downloadedChanged = Signal()           # background pre-fetch landed an installer
    quitForUpdate = Signal()               # app MUST quit so the helper can swap + relaunch

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._s = settings                         # SettingsController (channel / auto_download / notify_pre)
        self._checking = False
        self._available = False
        self._latest_tag = ""
        self._body = ""
        self._url = _RELEASES_PAGE
        self._last_checked = "never"
        self._check_error = ""                     # non-empty → the check couldn't complete
        self._pre_tag = ""                         # newer prerelease (notify-only, stable channel)
        self._pre_url = _RELEASES_PAGE
        self._result = None                       # written off-thread, drained on GUI thread
        self._ticks = 0                            # drain ticks → bound the wait
        self._poll = QTimer(self)
        self._poll.setInterval(120)
        self._poll.timeout.connect(self._drain)
        self._MAX_TICKS = 100                      # ~12 s before giving up on a hung fetch

        # ── background pre-fetch (updates/auto_download) ──────────────────────
        self._prefetching = False
        self._prefetched_path = ""                 # verified installer staged in the background
        self._prefetched_tag = ""                  # the release tag it belongs to
        self._pf_result = None                     # written off-thread, drained on GUI thread
        self._pf_poll = QTimer(self)
        self._pf_poll.setInterval(200)
        self._pf_poll.timeout.connect(self._drain_prefetch)

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

    # ── update-channel preferences (read from settings, safe if unset) ────
    def _channel(self):
        if self._s is None:
            return "Stable"
        try:
            return self._s.get_str("updates/channel", "Stable") or "Stable"
        except Exception:
            return "Stable"

    def _include_pre(self):
        """The install channel includes prereleases for anything that isn't the
        Stable channel (Pre-release, or a stale 'Nightly' from before the option
        was retired)."""
        return self._channel() != "Stable"

    def _notify_pre(self):
        if self._s is None:
            return False
        try:
            return bool(self._s.get_bool("updates/notify_pre", False))
        except Exception:
            return False

    def _auto_download(self):
        if self._s is None:
            return False
        try:
            return bool(self._s.get_bool("updates/auto_download", False))
        except Exception:
            return False

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

    # ── pre-release notice (notify-only; not the install target) ──────────
    @Property(bool, notify=changed)
    def prereleaseAvailable(self):
        return bool(self._pre_tag)

    @Property(str, notify=changed)
    def prereleaseTag(self):
        return self._pre_tag

    @Property(str, notify=changed)
    def prereleaseUrl(self):
        return self._pre_url

    # ── background pre-fetch: a verified installer is staged & ready ───────
    @Property(bool, notify=downloadedChanged)
    def updateDownloaded(self):
        return bool(self._prefetched_path
                    and self._prefetched_tag
                    and self._prefetched_tag == self._latest_tag
                    and os.path.exists(self._prefetched_path))

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

        include_pre = self._include_pre()
        notify_pre = self._notify_pre()

        def _work():
            try:
                from firefly import updater
                # The releases LIST (not /latest) so the channel can see betas.
                releases = updater.fetch_releases(_RELEASES_API)
                if isinstance(releases, dict) and releases.get("_rate_limited"):
                    payload = {"ok": False,
                               "err": "GitHub rate limit reached on this network — try again later."}
                elif not releases:
                    payload = {"ok": False,
                               "err": "Couldn't reach GitHub — check your network/proxy."}
                else:
                    chosen = updater.pick_release(releases, include_prerelease=include_pre)
                    rel = updater.parse_release(chosen) if chosen else {}
                    newer = bool(rel.get("tag")) and updater.is_newer(rel["tag"], __version__)
                    # notify-about-pre-releases (stable channel only): surface the
                    # newest prerelease if it's newer than both the current build
                    # and the stable release we'd install.  Notify-only — it never
                    # becomes the install target, so a Stable user stays on stable.
                    pre = {"tag": "", "url": ""}
                    if notify_pre and not include_pre:
                        pchosen = updater.pick_release(releases, include_prerelease=True)
                        prel = updater.parse_release(pchosen) if pchosen else {}
                        ptag = prel.get("tag") or ""
                        if (ptag and updater.is_newer(ptag, __version__)
                                and (not rel.get("tag") or updater.is_newer(ptag, rel["tag"]))):
                            pre = {"tag": ptag, "url": prel.get("html_url") or _RELEASES_PAGE}
                    payload = {"ok": True, "rel": rel, "newer": newer, "pre": pre}
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
            pre = res.get("pre") or {}
            self._pre_tag = pre.get("tag") or ""
            self._pre_url = pre.get("url") or _RELEASES_PAGE
            self.changed.emit()
            # Background download (updates/auto_download): silently pre-fetch the
            # verified installer so the user only has to click "Restart & install".
            if self._available and self._auto_download():
                self._start_prefetch()
            return
        else:
            # A failed fetch must NOT masquerade as "up to date".
            self._check_error = res.get("err") or "Couldn't check for updates."
            self._last_checked = "check failed"
        self.changed.emit()

    @Slot()
    def openReleasePage(self):
        QDesktopServices.openUrl(QUrl(self._url))

    @Slot()
    def openPrereleasePage(self):
        QDesktopServices.openUrl(QUrl(self._pre_url or _RELEASES_PAGE))

    # ── background pre-fetch (updates/auto_download) ───────────────────────
    def _start_prefetch(self):
        """Silently download + verify the channel's installer in the background
        so a later install is instant.  Best-effort: only in a frozen build (a
        source checkout can't install one anyway), never more than once per tag,
        and any failure is swallowed — the user can still click Download &
        install, which surfaces the real error visibly."""
        from firefly import updater
        if self._prefetching or not updater.is_frozen():
            return
        if self._prefetched_path and self._prefetched_tag == self._latest_tag:
            return                                  # already staged for this release
        self._prefetching = True
        self._pf_result = None
        self._pf_poll.start()
        include_pre = self._include_pre()

        def _work():
            try:
                from firefly import updater
                releases = updater.fetch_releases(_RELEASES_API)
                chosen = (updater.pick_release(releases, include_prerelease=include_pre)
                          if isinstance(releases, list) else None)
                rel = updater.parse_release(chosen) if chosen else {}
                asset = rel.get("asset")
                if not asset:
                    result = {"ok": False}
                else:
                    path = updater.download_asset(asset)   # verified; silent (no progress cb)
                    result = {"ok": True, "path": path, "tag": rel.get("tag") or self._latest_tag}
            except Exception:
                result = {"ok": False}
            try:    self._pf_result = result
            except Exception: pass

        threading.Thread(target=_work, daemon=True).start()

    def _drain_prefetch(self):
        if self._pf_result is None:
            return
        self._pf_poll.stop()
        res, self._pf_result = self._pf_result, None
        self._prefetching = False
        if res.get("ok"):
            self._prefetched_path = res.get("path") or ""
            self._prefetched_tag = res.get("tag") or ""
            self.downloadedChanged.emit()

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

        # Use the background-prefetched installer if it matches the release we're
        # about to install; otherwise download the channel's release on demand.
        prefetched = (self._prefetched_path
                      if (self._prefetched_path
                          and self._prefetched_tag == self._latest_tag
                          and os.path.exists(self._prefetched_path))
                      else "")
        include_pre = self._include_pre()

        def _work():
            try:
                from firefly import updater

                def _prog(done, total):
                    self._inst_progress = (done / total) if total else -1.0

                def _status(msg):
                    self._inst_status = str(msg)

                def _cancel():
                    return self._inst_cancel

                if prefetched:
                    self._inst_status = "Installing…"
                    path = prefetched
                else:
                    # The releases LIST (not /latest) so we install what the
                    # channel offered, prereleases included.
                    releases = updater.fetch_releases(_RELEASES_API)
                    chosen = (updater.pick_release(releases, include_prerelease=include_pre)
                              if isinstance(releases, list) else None)
                    rel = updater.parse_release(chosen) if chosen else {}
                    asset = rel.get("asset")
                    if not asset:
                        raise RuntimeError(
                            "The latest release has no installer for your platform. "
                            "Use 'Release notes' to download it manually.")
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
