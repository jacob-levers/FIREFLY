"""Unit tests for the in-app updater's pure logic (no network, no GUI).

The real mount/swap/relaunch can only be exercised on a frozen build per
OS (CI produces the artifacts); here we test everything that's pure:
version comparison, OS asset selection, release parsing, app-bundle-path
resolution, helper-script generation, download validation, and the
frozen-gated guards.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from firefly import updater as u


# ── version comparison ──────────────────────────────────────────────────
def test_parse_version_basic():
    assert u.parse_version("v2.41.0") == (2, 41, 0)
    assert u.parse_version("2.41") == (2, 41, 0)            # padded to 3
    assert u.parse_version("v2.41.0-dev3") == (2, 41, 0)    # suffix dropped
    assert u.parse_version("") == (0, 0, 0)


def test_is_newer():
    assert u.is_newer("v2.41.0", "2.40.0") is True
    assert u.is_newer("2.40.0", "2.40.0") is False
    assert u.is_newer("2.39.9", "2.40.0") is False
    assert u.is_newer("v2.41.1", "v2.41.0") is True


# ── OS asset selection ──────────────────────────────────────────────────
def test_current_os_asset_name(monkeypatch):
    monkeypatch.setattr(u.sys, "platform", "darwin")
    assert u.current_os_asset_name() == "FIREFLY-macOS.dmg"
    monkeypatch.setattr(u.sys, "platform", "win32")
    assert u.current_os_asset_name() == "FIREFLY-Windows.exe"
    monkeypatch.setattr(u.sys, "platform", "linux")
    assert u.current_os_asset_name() is None


_REL = {
    "tag_name": "v2.41.0",
    "html_url": "https://github.com/jacob-levers/FIREFLY/releases/tag/v2.41.0",
    "body": "- new in-app updater\n- bug fixes",
    "assets": [
        {"name": "FIREFLY-macOS.dmg",
         "browser_download_url": "https://x/dmg", "size": 123456},
        {"name": "FIREFLY-Windows.exe",
         "browser_download_url": "https://x/exe", "size": 789},
    ],
}


def test_select_asset():
    a = u.select_asset(_REL, "FIREFLY-macOS.dmg")
    assert a == {"name": "FIREFLY-macOS.dmg",
                 "url": "https://x/dmg", "size": 123456, "digest": ""}
    assert u.select_asset(_REL, "nope.bin") is None
    assert u.select_asset(_REL, None) is None
    assert u.select_asset({}, "FIREFLY-macOS.dmg") is None
    # GitHub's content digest is captured (used to verify the download).
    rel = {"assets": [{"name": "x.exe", "browser_download_url": "https://x",
                       "size": 9, "digest": "sha256:abc"}]}
    assert u.select_asset(rel, "x.exe")["digest"] == "sha256:abc"


def test_parse_release(monkeypatch):
    monkeypatch.setattr(u.sys, "platform", "darwin")
    pr = u.parse_release(_REL)
    assert pr["tag"] == "v2.41.0"
    assert pr["body"].startswith("- new in-app updater")
    assert pr["html_url"].endswith("v2.41.0")
    assert pr["asset"]["name"] == "FIREFLY-macOS.dmg"


def test_parse_release_empty():
    assert u.parse_release({}) == {
        "tag": "", "html_url": "", "body": "", "asset": None}


# ── bundle-path resolution ──────────────────────────────────────────────
def test_current_app_bundle_path():
    exe = "/Applications/FIREFLY.app/Contents/MacOS/FIREFLY"
    assert u.current_app_bundle_path(exe) == "/Applications/FIREFLY.app"
    assert u.current_app_bundle_path("/usr/local/bin/python3") is None


# ── helper-script generation (pure string builders) ─────────────────────
def test_macos_helper_script():
    s = u.macos_helper_script(4321, "/u/FIREFLY-macOS.dmg",
                              "/Applications/FIREFLY.app", "/u/relaunch.log")
    assert 'PID="4321"' in s
    assert 'kill -0 "$PID"' in s
    assert "hdiutil attach" in s
    assert "xattr -dr com.apple.quarantine" in s
    assert 'open "$TARGET"' in s
    assert "kill -9" in s            # force-kill fallback if the app hangs


def test_windows_helper_script():
    s = u.windows_helper_script(4321, r"C:\u\FIREFLY-Windows.exe",
                                r"C:\app\FIREFLY.exe", r"C:\u\relaunch.log")
    assert 'set "PID=4321"' in s
    assert 'tasklist /FI "PID eq %PID%"' in s
    assert "taskkill /F /T /PID %PID%" in s   # force-kill if the app hangs
    assert "copy /Y" in s
    assert 'start "" "%TARGET%"' in s
    assert 'del "%~f0"' in s
    # Hardened swap: back up the old exe, verify the copy's size, roll back on
    # failure (parity with the macOS path — a bad update must be recoverable).
    assert 'set "BACKUP=%TARGET%.bak"' in s
    assert 'copy /Y "%TARGET%" "%BACKUP%"' in s                       # backup first
    assert 'for %%A in ("%NEWEXE%") do set "SRCSIZE=%%~zA"' in s      # source size
    assert 'for %%A in ("%TARGET%") do set "DSTSIZE=%%~zA"' in s      # copied size
    assert 'if not "!SRCSIZE!"=="!DSTSIZE!"' in s                     # verify match
    assert 'copy /Y "%BACKUP%" "%TARGET%"' in s                       # restore on fail
    # Content check: SHA-256 of the copy must match the source (catches a
    # size-preserving corruption — the "decompression -3" cause).
    assert 'certutil -hashfile "%NEWEXE%" SHA256' in s
    assert 'certutil -hashfile "%TARGET%" SHA256' in s
    assert 'if /i not "!SRCHASH!"=="!DSTHASH!"' in s
    # Relaunch handshake: pause for AV, wait for the app's ready marker, and
    # kill + relaunch once if the first launch's extraction lost the race.
    assert 'set "SPTPALM_READY_MARKER=%MARKER%"' in s
    assert 'if exist "%MARKER%" goto ready' in s
    assert 'taskkill /F /IM "!TIMG!"' in s
    assert 'if !LAUNCHN! LSS 2' in s
    # On a clean, verified relaunch the backup is removed (no clutter); it's the
    # ``:ready`` branch, distinct from the give-up branch which keeps it.
    assert 'del "%BACKUP%"' in s
    assert ":ready" in s


def test_download_asset_integrity_failure_message(monkeypatch, tmp_path):
    """A persistent SHA-256 mismatch surfaces a clear 'integrity / install
    manually' error — not the misleading generic validation message."""
    import firefly.net_download as nd
    cr = tmp_path / "FIREFLY" / "crash_reports"
    cr.mkdir(parents=True)
    monkeypatch.setattr(u.crash_reporter, "crash_report_dir", lambda: str(cr))
    monkeypatch.setattr(u, "_validate_download", lambda p: True)   # skip format
    asset = {"name": "X.exe", "url": "https://x", "size": 10,
             "digest": "sha256:" + "a" * 64}                       # won't match

    def _fake_dl(url, dest, *, validate_cb=None, **kw):
        with open(dest, "wb") as fh:
            fh.write(b"some real bytes")
        if validate_cb:
            validate_cb(dest)            # hash mismatch → flags hash_failed
        raise nd.DownloadError("validation failed")
    monkeypatch.setattr(nd, "download_file", _fake_dl)

    with pytest.raises(u.UpdaterError) as ei:
        u.download_asset(asset)
    msg = str(ei.value).lower()
    assert "integrity" in msg and "manually" in msg


def test_sha256_and_digest_verification(tmp_path):
    """Download integrity: SHA-256 is computed and matched against GitHub's
    'sha256:HEX' digest; a wrong digest is rejected, a missing one passes."""
    import hashlib
    data = b"firefly-bytes" * 4096
    f = tmp_path / "blob.bin"
    f.write_bytes(data)
    want = hashlib.sha256(data).hexdigest()
    assert u._sha256_file(str(f)) == want.lower()
    assert u._digest_matches(str(f), "sha256:" + want) is True
    assert u._digest_matches(str(f), "sha256:" + "0" * 64) is False   # wrong → reject
    assert u._digest_matches(str(f), "") is True                       # no digest → pass
    assert u._digest_matches(str(f), "sha256:short") is True           # malformed → pass


# ── download validation (pure for the windows PE check) ─────────────────
def test_validate_windows_exe(tmp_path):
    ok = tmp_path / "a.exe"
    ok.write_bytes(b"MZ" + b"\x00" * 1_000_001)
    assert u._looks_like_windows_exe(str(ok)) is True
    small = tmp_path / "b.exe"
    small.write_bytes(b"MZ")                       # too small
    assert u._looks_like_windows_exe(str(small)) is False
    notpe = tmp_path / "c.exe"
    notpe.write_bytes(b"PK" + b"\x00" * 1_000_001)  # wrong magic
    assert u._looks_like_windows_exe(str(notpe)) is False


# ── staging dir + helper writing (app-data redirected to tmp) ───────────
def _redirect_appdata(monkeypatch, tmp_path):
    cr = tmp_path / "FIREFLY" / "crash_reports"
    cr.mkdir(parents=True)
    monkeypatch.setattr(u.crash_reporter, "crash_report_dir", lambda: str(cr))


def test_updates_dir(monkeypatch, tmp_path):
    _redirect_appdata(monkeypatch, tmp_path)
    d = u.updates_dir()
    assert d == str(tmp_path / "FIREFLY" / "updates")
    assert os.path.isdir(d)


def test_write_helper_executable_flag(monkeypatch, tmp_path):
    _redirect_appdata(monkeypatch, tmp_path)
    p = u._write_helper("firefly_update.sh", "#!/bin/bash\necho hi\n", True)
    assert os.path.isfile(p) and os.access(p, os.X_OK)
    p2 = u._write_helper("firefly_update.bat", "echo hi\n", False)
    assert os.path.isfile(p2)


# ── frozen-gated behaviour ──────────────────────────────────────────────
def test_is_frozen_default_false():
    # The test runner is never a frozen build.
    assert u.is_frozen() is False


def test_apply_update_not_frozen_raises(monkeypatch, tmp_path):
    _redirect_appdata(monkeypatch, tmp_path)
    f = tmp_path / "x.dmg"
    f.write_bytes(b"0" * 16)
    with pytest.raises(u.UpdaterError):
        u.apply_update(str(f))


def test_download_asset_without_asset_raises():
    with pytest.raises(u.UpdaterError):
        u.download_asset(None)
    with pytest.raises(u.UpdaterError):
        u.download_asset({"name": "x", "url": ""})


# ── _UpdateCheckThread delegates version-compare to updater ──────────────
def test_update_check_thread_delegates_parse_version():
    try:
        from firefly.ui.ui_widgets import _UpdateCheckThread
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"Qt UI import unavailable: {exc}")
    for tag in ("v2.41.0", "2.40.0-dev1", "1.2", ""):
        assert _UpdateCheckThread._parse_version(tag) == u.parse_version(tag)
