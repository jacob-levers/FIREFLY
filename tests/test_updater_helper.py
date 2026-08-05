"""Updater staging, validation, and detached-helper regressions."""
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from firefly import updater


def test_windows_helper_keeps_no_backup():
    s = updater.windows_helper_script(
        4321, r"C:\up\FIREFLY-Windows.exe", r"C:\app\FIREFLY.exe",
        r"C:\up\relaunch.log")
    # No backup of the old exe anywhere.
    assert "BACKUP" not in s
    assert ".bak" not in s
    # Stage next to the target → verify → swap by rename.
    assert 'set "STAGED=%TARGET%.new"' in s
    assert ":stage" in s and ":swap" in s
    assert 'move /Y "%STAGED%" "%TARGET%"' in s
    # Integrity check + relaunch handshake are still present.
    assert "certutil -hashfile" in s
    assert 'start "" "%TARGET%"' in s
    assert "firefly_relaunch_ok.marker" in s


def test_windows_helper_reads_paths_positionally():
    # Paths come from %~1..%~4 (never interpolated), so a '%' in a path is safe.
    s = updater.windows_helper_script(1, "N", "T", "L")
    assert 'set "PID=%~1"' in s
    assert 'set "NEWEXE=%~2"' in s
    assert 'set "TARGET=%~3"' in s
    assert 'set "LOG=%~4"' in s


def test_macos_helper_unchanged_still_backs_up():
    # The macOS path is deliberately untouched — it still uses a temp backup.
    s = updater.macos_helper_script(1, "/tmp/x.dmg", "/Applications/FIREFLY.app",
                                    "/tmp/log")
    assert "BACKUP" in s


def test_arm64_macos_accepts_only_architecture_explicit_asset(monkeypatch):
    monkeypatch.setattr(updater, "is_macos", lambda: True)
    monkeypatch.setattr(updater, "is_windows", lambda: False)
    monkeypatch.setattr(updater.platform, "machine", lambda: "arm64")
    assert updater.current_os_asset_names() == ("FIREFLY-macOS-arm64.dmg",)
    release = {"tag_name": "v1.0.0", "assets": [
        {"name": "FIREFLY-macOS.dmg", "browser_download_url": "legacy"},
        {"name": "FIREFLY-macOS-arm64.dmg", "browser_download_url": "arm"},
    ]}
    assert updater.parse_release(release)["asset"]["name"] == "FIREFLY-macOS-arm64.dmg"


def test_intel_macos_has_no_arm64_installer(monkeypatch):
    monkeypatch.setattr(updater, "is_macos", lambda: True)
    monkeypatch.setattr(updater, "is_windows", lambda: False)
    monkeypatch.setattr(updater.platform, "machine", lambda: "x86_64")
    assert updater.current_os_asset_name() is None
    assert "Apple Silicon" in updater.installer_unavailable_message()
    assert "Intel" in updater.installer_unavailable_message()


# ── DMG validation must not hang the updater (100%-then-stuck bug) ────────────
def test_non_udif_dmg_is_rejected_without_spawning_hdiutil(tmp_path, monkeypatch):
    """Malformed bytes fail the bounded local check without starting the flaky
    post-100% ``hdiutil imageinfo`` subprocess."""
    big = tmp_path / "FIREFLY-macOS.dmg"
    big.write_bytes(b"\0" * 2_000_000)             # > 1 MB size gate

    def must_not_run(*args, **kwargs):
        raise AssertionError("DMG sanity check spawned hdiutil")

    monkeypatch.setattr(updater.subprocess, "run", must_not_run)
    assert updater._looks_like_dmg(str(big)) is False


def test_valid_udif_dmg_is_checked_locally_after_download(tmp_path, monkeypatch):
    """A complete UDIF image must not invoke ``hdiutil`` after progress reaches
    100%.  That subprocess can hang/fail transiently; the downloader interprets
    either result as corrupt bytes and starts the entire transfer again.

    FIREFLY verifies the GitHub SHA-256 separately, so the local format gate only
    needs the deterministic UDIF ``koly`` trailer check.
    """
    dmg = tmp_path / "FIREFLY-macOS-arm64.dmg.part"
    dmg.write_bytes(b"x" * 1_500_000 + b"koly" + b"\0" * 508)

    def must_not_run(*args, **kwargs):
        raise AssertionError("post-download validation spawned hdiutil")

    monkeypatch.setattr(updater.subprocess, "run", must_not_run)
    assert updater._looks_like_dmg(str(dmg)) is True


def test_concurrent_requests_share_one_staged_asset(tmp_path, monkeypatch):
    """Auto-prefetch and a user click can overlap.  They target the exact same
    ``.part``/installer path, so two downloader instances corrupt each other's
    state and visibly reach 100%, error, then download again.  Both callers must
    share the first verified transfer instead.
    """
    payload = b"verified installer bytes"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    asset = {"name": "FIREFLY-test.bin", "url": "https://x/asset",
             "size": len(payload), "digest": digest}
    monkeypatch.setattr(updater, "updates_dir", lambda: str(tmp_path))
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)

    guard = threading.Lock()
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = {"n": 0, "active": 0, "max_active": 0}

    def fake_download(url, dest, **kwargs):
        with guard:
            calls["n"] += 1
            calls["active"] += 1
            calls["max_active"] = max(calls["max_active"], calls["active"])
            if calls["n"] == 1:
                first_entered.set()
        release_first.wait(2)
        Path(dest).write_bytes(payload)
        with guard:
            calls["active"] -= 1

    monkeypatch.setattr(updater.net_download, "download_file", fake_download)
    results = []
    errors = []

    def request():
        try:
            results.append(updater.download_asset(asset))
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    first = threading.Thread(target=request)
    second = threading.Thread(target=request)
    first.start()
    assert first_entered.wait(1)
    second.start()
    time.sleep(0.08)  # enough for the unguarded second call to enter downloader
    release_first.set()
    first.join(2)
    second.join(2)

    assert not errors
    assert len(results) == 2
    assert calls["n"] == 1
    assert calls["max_active"] == 1


def test_partial_without_matching_asset_identity_is_not_resumed(tmp_path, monkeypatch):
    """Release assets reuse a constant filename.  A ``.part`` left by version N
    must not become the prefix of version N+1; that only discovers the mismatch
    after reaching 100%, then deletes the file and downloads it all over again.
    """
    payload = b"new release installer"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    asset = {"name": "FIREFLY-test.bin", "url": "https://x/v-next/asset",
             "size": len(payload), "digest": digest}
    dest = tmp_path / asset["name"]
    Path(str(dest) + ".part").write_bytes(b"old release prefix")
    monkeypatch.setattr(updater, "updates_dir", lambda: str(tmp_path))
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)
    saw_stale_partial = []

    def fake_download(url, out, **kwargs):
        saw_stale_partial.append(Path(out + ".part").exists())
        Path(out).write_bytes(payload)

    monkeypatch.setattr(updater.net_download, "download_file", fake_download)
    assert updater.download_asset(asset) == str(dest)
    assert saw_stale_partial == [False]


def test_partial_with_matching_asset_identity_remains_resumable(tmp_path):
    """The identity guard must preserve an interrupted transfer when it really
    belongs to the same URL/size/digest."""
    dest = tmp_path / "FIREFLY-test.bin"
    part = Path(str(dest) + ".part")
    part.write_bytes(b"current release prefix")
    identity = {"url": "https://x/current", "size": 456,
                "digest": "sha256:" + "c" * 64}
    sidecar = Path(updater._asset_identity_path(str(dest)))
    sidecar.write_text(json.dumps(identity), encoding="utf-8")

    updater._prepare_resume_state(str(dest), identity)

    assert part.read_bytes() == b"current release prefix"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == identity


def test_failed_stale_partial_cleanup_does_not_bless_new_identity(tmp_path,
                                                                 monkeypatch):
    """If an AV/file lock prevents deletion of old resume bytes, do not write a
    sidecar claiming those bytes belong to the new release."""
    dest = tmp_path / "FIREFLY-test.bin"
    part = Path(str(dest) + ".part")
    part.write_bytes(b"old release prefix")
    sidecar = Path(updater._asset_identity_path(str(dest)))
    old_identity = {"url": "https://x/old", "size": 99,
                    "digest": "sha256:" + "b" * 64}
    sidecar.write_text(json.dumps(old_identity), encoding="utf-8")
    identity = {"url": "https://x/new", "size": 123,
                "digest": "sha256:" + "a" * 64}
    real_remove = updater.os.remove

    def deny_part(path):
        if str(path) == str(part):
            raise PermissionError("locked by scanner")
        return real_remove(path)

    monkeypatch.setattr(updater.os, "remove", deny_part)
    monkeypatch.setattr(updater.time, "sleep", lambda *_: None)
    with pytest.raises(updater.UpdaterError):
        updater._prepare_resume_state(str(dest), identity)
    assert json.loads(sidecar.read_text(encoding="utf-8")) == old_identity
