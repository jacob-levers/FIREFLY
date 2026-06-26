"""Update-channel wiring: updater.pick_release selection + UpdatesController
respecting updates/channel, updates/notify_pre and updates/auto_download.

These are the controls in Preferences ▸ Updates that previously did nothing —
this guards that they actually drive the GitHub query + install flow now.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── pure release selection ────────────────────────────────────────────────────
def test_pick_release_stable_skips_prerelease_and_drafts():
    from firefly import updater
    rels = [
        {"tag_name": "v2.76.20", "prerelease": False, "draft": False},
        {"tag_name": "v2.77.0-rc1", "prerelease": True, "draft": False},
        {"tag_name": "v2.78.0", "prerelease": False, "draft": True},   # draft → ignored
        {"tag_name": "v2.76.19", "prerelease": False, "draft": False},
    ]
    assert updater.pick_release(rels, include_prerelease=False)["tag_name"] == "v2.76.20"
    assert updater.pick_release(rels, include_prerelease=True)["tag_name"] == "v2.77.0-rc1"


def test_pick_release_chooses_by_version_not_order():
    from firefly import updater
    rels = [
        {"tag_name": "v2.10.0", "prerelease": False, "draft": False},
        {"tag_name": "v2.9.0", "prerelease": False, "draft": False},
        {"tag_name": "v2.100.0", "prerelease": False, "draft": False},
    ]
    assert updater.pick_release(rels, include_prerelease=False)["tag_name"] == "v2.100.0"


def test_pick_release_empty_and_garbage():
    from firefly import updater
    assert updater.pick_release([], False) is None
    assert updater.pick_release("nope", True) is None
    assert updater.pick_release([{"tag_name": "v1", "draft": True}], True) is None


# ── controller honours the channel / notify-pre settings ──────────────────────
class _FakeSettings:
    def __init__(self, **d): self.d = d
    def get_str(self, k, default=""): return self.d.get(k, default)
    def get_bool(self, k, default=False): return self.d.get(k, default)


_FAKE_RELEASES = [
    {"tag_name": "v2.99.0", "prerelease": False, "draft": False,
     "html_url": "https://x/stable", "body": "stable notes"},
    {"tag_name": "v3.0.0-rc1", "prerelease": True, "draft": False,
     "html_url": "https://x/beta", "body": "beta notes"},
]


@pytest.fixture
def mk_controller(monkeypatch):
    """Factory for UpdatesControllers that tears each one down deterministically
    (stop its QTimers + delete the QObject while the QApplication is still alive),
    so lingering timers never segfault at interpreter shutdown."""
    from firefly.ui.controllers import updates_controller as uc
    created = []

    def _make(settings):
        c = uc.UpdatesController(settings)
        created.append(c)
        return c

    yield _make

    for c in created:
        for timer in (c._poll, c._inst_poll, c._pf_poll):
            try:
                timer.stop()
            except Exception:
                pass
        c.setParent(None)
        c.deleteLater()
    _app.processEvents()


def _run_check(c):
    c.checkNow()
    t0 = time.time()
    while c.checking and time.time() - t0 < 5:
        _app.processEvents()
        time.sleep(0.01)


def test_stable_channel_ignores_prerelease(monkeypatch, mk_controller):
    from firefly import updater
    monkeypatch.setattr(updater, "fetch_releases", lambda url, timeout=6.0: list(_FAKE_RELEASES))
    c = mk_controller(_FakeSettings(**{"updates/channel": "Stable"}))
    _run_check(c)
    assert c.updateAvailable is True
    assert c.latestTag == "v2.99.0"
    assert c.prereleaseTag == ""              # notify-pre off → no beta surfaced


def test_notify_pre_surfaces_beta_without_changing_install_target(monkeypatch, mk_controller):
    from firefly import updater
    monkeypatch.setattr(updater, "fetch_releases", lambda url, timeout=6.0: list(_FAKE_RELEASES))
    c = mk_controller(_FakeSettings(**{"updates/channel": "Stable", "updates/notify_pre": True}))
    _run_check(c)
    assert c.latestTag == "v2.99.0"           # install target stays stable
    assert c.prereleaseAvailable is True
    assert c.prereleaseTag == "v3.0.0-rc1"    # beta surfaced separately


def test_prerelease_channel_installs_the_beta(monkeypatch, mk_controller):
    from firefly import updater
    monkeypatch.setattr(updater, "fetch_releases", lambda url, timeout=6.0: list(_FAKE_RELEASES))
    c = mk_controller(_FakeSettings(**{"updates/channel": "Pre-release"}))
    _run_check(c)
    assert c.latestTag == "v3.0.0-rc1"
    assert c.prereleaseTag == ""              # it's the install target, not a side-note


def test_no_settings_defaults_to_stable(monkeypatch, mk_controller):
    from firefly import updater
    monkeypatch.setattr(updater, "fetch_releases", lambda url, timeout=6.0: list(_FAKE_RELEASES))
    c = mk_controller(None)                   # headless / no settings → safe default
    _run_check(c)
    assert c.latestTag == "v2.99.0"


# ── background auto-download: settings read + the install-reuse condition ──────
# NB: the live prefetch/install paths spawn real download/install threads which,
# combined with QML-engine teardown elsewhere in the full suite, trip a known
# offscreen-Qt fault.  They're exercised manually; here we test the pure logic —
# the settings gate and the exact (path + tag + exists) condition that decides
# whether an install reuses the staged file — without spawning any thread.
def test_auto_download_setting_gate(mk_controller):
    on = mk_controller(_FakeSettings(**{"updates/auto_download": True}))
    off = mk_controller(_FakeSettings())
    none = mk_controller(None)
    assert on._auto_download() is True
    assert off._auto_download() is False     # default off
    assert none._auto_download() is False     # headless / no settings → safe default


def test_update_downloaded_reuse_condition(mk_controller, tmp_path):
    """updateDownloaded gates on a staged file whose tag matches the offered
    release AND still exists — the same condition downloadAndInstall uses to skip
    the download and install the prefetched file."""
    staged = tmp_path / "FIREFLY-installer"
    staged.write_bytes(b"x" * 32)
    c = mk_controller(_FakeSettings())
    c._latest_tag = "v2.99.0"

    assert c.updateDownloaded is False                 # nothing staged yet

    c._prefetched_tag = "v2.99.0"
    c._prefetched_path = str(staged)
    assert c.updateDownloaded is True                  # matches + exists → reuse

    c._prefetched_tag = "v2.98.0"                       # staged a different release
    assert c.updateDownloaded is False

    c._prefetched_tag = "v2.99.0"
    staged.unlink()                                    # staged file vanished
    assert c.updateDownloaded is False
