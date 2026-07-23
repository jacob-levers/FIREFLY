"""The Windows update helper keeps NO backup .exe — it stages the new build next
to the target, verifies it, then renames it into place."""
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


# ── DMG validation must not hang the updater (100%-then-stuck bug) ────────────
def test_dmg_validation_times_out_instead_of_hanging(tmp_path, monkeypatch):
    """`hdiutil imageinfo` runs after the download hits 100%.  With no timeout a
    hung hdiutil froze verification forever ('stays at 100% for ages').  A
    timeout must be passed, and a TimeoutExpired is treated as 'not a DMG'
    (fail-closed → retry / manual install) rather than blocking."""
    import subprocess
    big = tmp_path / "FIREFLY-macOS.dmg"
    big.write_bytes(b"\0" * 2_000_000)             # > 1 MB size gate

    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    assert updater._looks_like_dmg(str(big)) is False   # hang → fail, not block
    assert seen.get("timeout"), "hdiutil imageinfo must be given a timeout"
