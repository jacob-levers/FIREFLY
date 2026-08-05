"""Updater staging, validation, and detached-helper regressions."""
import hashlib
import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from firefly import updater


def _hold_file_lock(path, ready, release):
    """Spawn-safe helper proving the updater lock is kernel-visible."""
    lock = updater._InterProcessFileLock(path)
    if not lock.acquire():
        return
    ready.set()
    release.wait(5)
    lock.release()


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
    assert 'if /i not "%DSTHASH%"=="%EXPECTED_SHA%"' in s
    assert 'start "" "%TARGET%"' in s
    assert "firefly_relaunch_ok.marker" in s


def test_windows_helper_reads_paths_from_dedicated_environment():
    # Dynamic values are never batch arguments, where cmd metacharacters would
    # be reparsed before the script starts.
    s = updater.windows_helper_script(1, "N", "T", "L")
    assert 'set "PID=%FIREFLY_UPDATE_PID%"' in s
    assert 'set "NEWEXE=%FIREFLY_UPDATE_SOURCE%"' in s
    assert 'set "TARGET=%FIREFLY_UPDATE_TARGET%"' in s
    assert 'set "LOG=%FIREFLY_UPDATE_LOG%"' in s
    assert 'set "EXPECTED_SIZE=%FIREFLY_UPDATE_EXPECTED_SIZE%"' in s
    assert 'set "EXPECTED_SHA=%FIREFLY_UPDATE_EXPECTED_SHA%"' in s
    assert "source verified against GitHub metadata" in s


def test_windows_helper_preserves_bang_in_paths():
    """Delayed expansion strips ``!`` characters from expanded path values.
    Keep it disabled and avoid all ``!VAR!`` expansion in the helper.
    """
    s = updater.windows_helper_script(
        1, r"C:\Users\Lab!One\new.exe", r"C:\Apps\Fire!fly\FIREFLY.exe",
        r"C:\Users\Lab!One\update.log")
    assert "setlocal DisableDelayedExpansion" in s
    assert "enabledelayedexpansion" not in s.lower()
    assert "!NEWEXE!" not in s
    assert "!TARGET!" not in s
    assert "!LOG!" not in s


def test_windows_helper_deletes_source_only_after_ready_success():
    """Content-addressed EXEs must not accumulate after successful updates,
    while failures and ready-marker timeouts retain the installer for repair.
    """
    s = updater.windows_helper_script(1, "N", "T", "L")
    ready = s.index("\n:ready\n")
    cleanup = s.index('del "%NEWEXE%"')
    end = s.index("\n:end\n", ready)
    assert ready < cleanup < end
    assert 'del "%NEWEXE%"' not in s[:ready]


def test_macos_helper_unchanged_still_backs_up():
    # The macOS path is deliberately untouched — it still uses a temp backup.
    s = updater.macos_helper_script(1, "/tmp/x.dmg", "/Applications/FIREFLY.app",
                                    "/tmp/log")
    assert "BACKUP" in s
    assert 'EXPECTED_SIZE="$5"' in s
    assert 'EXPECTED_SHA="$6"' in s
    assert "/usr/bin/shasum -a 256" in s


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


def test_same_published_filename_cannot_replace_another_release(tmp_path,
                                                                monkeypatch):
    """Two releases can publish the same asset filename.  Their verified files
    must remain distinct so release B cannot replace bytes already handed to
    release A's detached installer.
    """
    payloads = {
        "https://x/v1/asset": b"release one bytes",
        "https://x/v2/asset": b"release two bytes",
    }
    assets = []
    for url, payload in payloads.items():
        assets.append({
            "name": "FIREFLY-same-name.bin",
            "url": url,
            "size": len(payload),
            "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        })
    monkeypatch.setattr(updater, "updates_dir", lambda: str(tmp_path))
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)

    def fake_download(url, out, **kwargs):
        Path(out).write_bytes(payloads[url])

    monkeypatch.setattr(updater.net_download, "download_file", fake_download)
    first = updater.download_asset(assets[0])
    second = updater.download_asset(assets[1])

    assert first != second
    assert Path(first).read_bytes() == payloads[assets[0]["url"]]
    assert Path(second).read_bytes() == payloads[assets[1]["url"]]
    assert Path(first).parent.name == assets[0]["digest"].split(":", 1)[1]
    assert Path(second).parent.name == assets[1]["digest"].split(":", 1)[1]


def test_asset_lock_is_visible_to_another_process(tmp_path):
    """A second FIREFLY process must not enter the same resume state while the
    first owns it; a thread-only lock would let this assertion fail.
    """
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    lock_path = str(tmp_path / "asset.lock")
    holder = context.Process(target=_hold_file_lock,
                             args=(lock_path, ready, release))
    holder.start()
    try:
        assert ready.wait(5), "child process did not acquire updater lock"
        contender = updater._InterProcessFileLock(lock_path)
        assert contender.acquire() is False
        contender.release()
    finally:
        release.set()
        holder.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(2)
    assert holder.exitcode == 0


def test_asset_lock_permission_failure_does_not_spin(tmp_path, monkeypatch):
    """A permanent lock-file error is actionable, not mistaken for another
    process holding the lock forever.
    """
    import builtins
    real_open = builtins.open
    lock_path = str(tmp_path / "denied.lock")

    def deny_lock(path, *args, **kwargs):
        if str(path) == lock_path:
            raise PermissionError("read-only update folder")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_lock)
    lock = updater._InterProcessFileLock(lock_path)
    with pytest.raises(updater.UpdaterError, match="Couldn't open"):
        lock.acquire()


def test_partial_without_matching_asset_identity_is_not_resumed(tmp_path, monkeypatch):
    """Release assets reuse a constant filename.  A ``.part`` left by version N
    must not become the prefix of version N+1; that only discovers the mismatch
    after reaching 100%, then deletes the file and downloads it all over again.
    """
    payload = b"new release installer"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    asset = {"name": "FIREFLY-test.bin", "url": "https://x/v-next/asset",
             "size": len(payload), "digest": digest}
    monkeypatch.setattr(updater, "updates_dir", lambda: str(tmp_path))
    dest = Path(updater._asset_staging_path(asset))
    Path(str(dest) + ".part").write_bytes(b"old release prefix")
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)
    saw_stale_partial = []

    def fake_download(url, out, **kwargs):
        saw_stale_partial.append(Path(out + ".part").exists())
        Path(out).write_bytes(payload)

    monkeypatch.setattr(updater.net_download, "download_file", fake_download)
    assert updater.download_asset(asset) == str(dest)
    assert saw_stale_partial == [False]


def test_verify_staged_asset_rejects_post_download_mutation(tmp_path, monkeypatch):
    payload = b"verified immutable installer"
    staged = tmp_path / "installer.bin"
    staged.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)

    updater.verify_staged_asset(
        str(staged), expected_size=len(payload), expected_digest=digest)
    staged.write_bytes(b"mutated immutable installer")
    with pytest.raises(updater.UpdaterError, match="SHA-256 changed"):
        updater.verify_staged_asset(
            str(staged), expected_size=staged.stat().st_size,
            expected_digest=digest)


def test_apply_passes_expected_metadata_to_detached_helper(tmp_path, monkeypatch):
    """The helper must independently verify the exact GitHub size/SHA after the
    running app exits, closing the final pre-swap mutation window.
    """
    payload = b"verified windows installer"
    special = tmp_path / "Lab! 50% & data"
    special.mkdir()
    staged = special / "FIREFLY-Windows.exe"
    staged.write_bytes(payload)
    target = special / "installed & stable!" / "FIREFLY.exe"
    target.parent.mkdir()
    target.write_bytes(b"old")
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    sha = digest.split(":", 1)[1]
    popen_calls = []

    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "is_macos", lambda: False)
    monkeypatch.setattr(updater, "is_windows", lambda: True)
    monkeypatch.setattr(updater, "updates_dir", lambda: str(special))
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)
    monkeypatch.setattr(updater.sys, "executable", str(target))
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    monkeypatch.setattr(updater, "_write_helper",
                        lambda name, content, executable: str(special / name))
    monkeypatch.setattr(updater.subprocess, "Popen",
                        lambda args, **kwargs: popen_calls.append((args, kwargs)))

    updater.apply_update(
        str(staged), expected_size=len(payload), expected_digest=digest)

    assert len(popen_calls) == 1
    command, kwargs = popen_calls[0]
    env = kwargs["env"]
    helper = str(special / f"firefly_update_{updater.os.getpid()}_{sha[:12]}.bat")
    assert command.endswith('/d /v:off /s /c ""%FIREFLY_UPDATE_HELPER%""')
    assert kwargs["executable"].lower().endswith("cmd.exe")
    assert str(staged) not in command
    assert str(target) not in command
    assert str(special) not in command
    assert sha not in command
    assert env["FIREFLY_UPDATE_HELPER"] == helper
    assert env["FIREFLY_UPDATE_SOURCE"] == str(staged)
    assert env["FIREFLY_UPDATE_TARGET"] == str(target)
    assert env["FIREFLY_UPDATE_LOG"] == str(special / "relaunch.log")
    assert env["FIREFLY_UPDATE_EXPECTED_SIZE"] == str(len(payload))
    assert env["FIREFLY_UPDATE_EXPECTED_SHA"] == sha


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
