"""
FIREFLY in-app updater (frozen DMG/EXE builds only).

Lets a packaged FIREFLY download a newer GitHub release and install it in
place, then relaunch — so lab users never have to re-download from the
Releases page by hand.  Pure stdlib + ``firefly.net_download``; NO PySide6
import, so it stays usable from any thread and is unit-testable headless.

Scope: only meaningful in a frozen build (PyInstaller ``.app`` /
one-file ``.exe``).  Running from source the public helpers no-op or
report "not frozen" so the UI can hide the feature.

Per-OS install model — a running app can't atomically replace its own
files, so ``apply_update()`` stages a tiny detached helper script that:
  * waits for THIS process (handed its PID) to exit,
  * swaps the new build into the current install location,
  * relaunches it,
  * deletes itself.
The Qt layer spawns that helper, then quits cleanly (its closeEvent
tears down the worker subprocess, napari/Metal, etc.).

macOS note: the app is unsigned/un-notarised, so the helper clears the
Gatekeeper quarantine flag (``xattr -dr com.apple.quarantine``) on the
freshly-copied bundle, mirroring the manual "right-click → Open" step.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional

from firefly import crash_reporter
from firefly import net_download


# ── Errors ────────────────────────────────────────────────────────────────────
class UpdaterError(RuntimeError):
    """A user-facing updater failure.  ``reveal_path`` (if set) is a folder
    the UI can open so the user can finish the install manually."""

    def __init__(self, message: str, reveal_path: Optional[str] = None):
        super().__init__(message)
        self.reveal_path = reveal_path


# ── Platform / environment ────────────────────────────────────────────────────
def is_frozen() -> bool:
    """True when running as a PyInstaller-frozen build (.app / .exe)."""
    return bool(getattr(sys, "frozen", False))


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_windows() -> bool:
    return sys.platform == "win32"


def current_os_asset_name() -> Optional[str]:
    """The GitHub release asset filename for this platform, or None on an
    unsupported OS (the updater only ships macOS + Windows builds)."""
    if is_macos():
        return "FIREFLY-macOS.dmg"
    if is_windows():
        return "FIREFLY-Windows.exe"
    return None


def updates_dir() -> str:
    """`<app-data>/FIREFLY/updates`, created on first use.  Staging area
    for the downloaded installer + the helper script + relaunch log."""
    d = os.path.join(os.path.dirname(crash_reporter.crash_report_dir()),
                     "updates")
    os.makedirs(d, exist_ok=True)
    return d


# ── Version comparison ────────────────────────────────────────────────────────
def parse_version(s: str) -> "tuple[int, ...]":
    """Parse a 'v2.41.0' / '2.41.0-dev3' style tag into a comparable tuple
    of ints.  Non-numeric suffix segments compare as 0.  (The single
    canonical comparator — ``_UpdateCheckThread`` delegates here.)"""
    import re
    s = (s or "").lstrip("vV").split("-", 1)[0]
    parts = []
    for chunk in s.split("."):
        m = re.match(r"(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def is_newer(latest: str, current: str) -> bool:
    """True if release tag ``latest`` is strictly newer than ``current``."""
    return parse_version(latest) > parse_version(current)


# ── GitHub release discovery ──────────────────────────────────────────────────
def fetch_latest_release(api_url: str, timeout: float = 6.0) -> dict:
    """GET the GitHub 'latest release' JSON.  Returns the parsed dict, or an
    empty dict on ANY failure (offline, bad JSON).

    On a GitHub API RATE LIMIT (HTTP 403/429 with X-RateLimit-Remaining: 0)
    returns a marker ``{"_rate_limited": True, "_reset": <epoch>}`` instead — the
    unauthenticated limit is only 60 requests/hour PER IP, which is shared by
    every device behind a NAT (e.g. a university network), so it's hit routinely
    and must NOT be reported as 'check your internet connection'."""
    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "FIREFLY-app"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read()
        data = json.loads(blob)
        return data if isinstance(data, dict) else {}
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            try:
                hdrs = exc.headers or {}
                rem = hdrs.get("X-RateLimit-Remaining")
                reset = hdrs.get("X-RateLimit-Reset")
                if rem == "0" or hdrs.get("Retry-After"):
                    try:
                        reset_epoch = int(reset) if reset else 0
                    except (TypeError, ValueError):
                        reset_epoch = 0
                    return {"_rate_limited": True, "_reset": reset_epoch}
            except Exception:
                pass
        return {}
    except Exception:
        return {}


def select_asset(release_json: dict,
                 asset_name: Optional[str]) -> Optional[dict]:
    """Find the named asset in a release JSON; return
    ``{"name", "url", "size", "digest"}`` or None if absent.  ``digest`` is the
    GitHub-provided content hash (e.g. ``"sha256:abcd…"``) used to verify the
    download was not corrupted in transit; "" if GitHub didn't supply one."""
    if not asset_name or not isinstance(release_json, dict):
        return None
    for a in release_json.get("assets", []) or []:
        if not isinstance(a, dict):
            continue
        if a.get("name") == asset_name:
            url = a.get("browser_download_url")
            if not url:
                return None
            return {"name": asset_name,
                    "url": url,
                    "size": int(a.get("size") or 0),
                    "digest": str(a.get("digest") or "")}
    return None


def parse_release(release_json: dict) -> dict:
    """Normalise a GitHub release JSON into the fields the UI needs:
    ``{"tag", "html_url", "body", "asset"}``.  ``asset`` is the installer
    for the current OS (or None if this platform has no matching asset)."""
    if not isinstance(release_json, dict):
        release_json = {}
    if release_json.get("_rate_limited"):
        return {"tag": "", "html_url": "", "body": "", "asset": None,
                "rate_limited": True,
                "rate_limit_reset": int(release_json.get("_reset") or 0)}
    return {
        "tag": release_json.get("tag_name") or "",
        "html_url": release_json.get("html_url") or "",
        "body": release_json.get("body") or "",
        "asset": select_asset(release_json, current_os_asset_name()),
    }


# ── Download + integrity ──────────────────────────────────────────────────────
def _looks_like_windows_exe(path: str) -> bool:
    """Cheap sanity check: a real PE starts with 'MZ' and isn't tiny."""
    try:
        if os.path.getsize(path) < 1_000_000:
            return False
        with open(path, "rb") as fh:
            return fh.read(2) == b"MZ"
    except Exception:
        return False


def _looks_like_dmg(path: str) -> bool:
    """True if hdiutil can read the image header (rejects an HTML error
    page or a truncated download)."""
    try:
        if os.path.getsize(path) < 1_000_000:
            return False
        rc = subprocess.run(["hdiutil", "imageinfo", path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
        return rc == 0
    except Exception:
        return False


def _validate_download(path: str) -> bool:
    """Format check for the downloaded asset, dispatched by OS.  Passed to
    net_download as ``validate_cb`` (runs before the atomic rename)."""
    if is_macos():
        return _looks_like_dmg(path)
    if is_windows():
        return _looks_like_windows_exe(path)
    return os.path.getsize(path) > 0 if os.path.exists(path) else False


def _sha256_file(path: str) -> Optional[str]:
    """Lower-case hex SHA-256 of ``path``, or None on error."""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(blk)
        return h.hexdigest().lower()
    except Exception:
        return None


def _is_verifiable_digest(digest: str) -> bool:
    """True only when ``digest`` is a usable ``"sha256:<64-hex>"`` we can
    actually authenticate the binary against.  A missing / malformed digest
    means we CANNOT verify the download, so the caller must fail closed."""
    d = (digest or "").strip().lower()
    if not d.startswith("sha256:"):
        return False
    want = d.split(":", 1)[1].strip()
    return len(want) == 64 and all(c in "0123456789abcdef" for c in want)


def _digest_matches(path: str, digest: str) -> bool:
    """True if ``path``'s SHA-256 matches GitHub's asset ``digest``
    (``"sha256:HEX"``).  Returns True when no usable digest is supplied — we
    can't verify what we weren't given, and the size/format checks still apply."""
    d = (digest or "").strip().lower()
    if not d.startswith("sha256:"):
        return True
    want = d.split(":", 1)[1].strip()
    if len(want) != 64:
        return True
    got = _sha256_file(path)
    return got is not None and got == want


def download_asset(asset: dict,
                   *,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   cancel_cb: Optional[Callable[[], bool]] = None,
                   status_cb: Optional[Callable[[str], None]] = None) -> str:
    """Download ``asset`` (from ``select_asset``/``parse_release``) into
    the updates staging dir.  Returns the path to the verified file.
    Raises ``UpdaterError`` on failure.

    Uses a generous retry budget (6 attempts, backoff out to ~30 s): a
    freshly-published GitHub asset can have its download edge return HTTP
    504 in bursts for a minute or two, and we want the update to ride that
    out rather than fail the user."""
    if not asset or not asset.get("url"):
        raise UpdaterError("No installer is available for your platform.")
    dest = os.path.join(updates_dir(), asset["name"])
    digest = str(asset.get("digest") or "")
    # Fail CLOSED when GitHub didn't publish a verifiable SHA-256: the only other
    # checks are "size > 1 MB and starts with MZ", which any substituted or
    # transit-corrupted PE passes.  The installer is unsigned and gets atomically
    # swapped in + relaunched, so an unauthenticated binary must NOT be installed
    # automatically — route the user to the Releases page to download by hand. (#20)
    # The received digest is echoed so a future GitHub digest-format change (which
    # would otherwise silently disable auto-update for everyone) is diagnosable
    # from the log instead of looking like "this release has no checksum".  (R2-10)
    if not _is_verifiable_digest(digest):
        _seen = repr(digest) if digest else "(none provided)"
        raise UpdaterError(
            "FIREFLY can't verify this download is authentic — the release asset "
            f"has no usable SHA-256 checksum (got digest={_seen}). Nothing was "
            "installed; download the installer manually from the Releases page "
            "instead. (If auto-update suddenly stopped working for everyone, the "
            "GitHub asset-digest format may have changed.)",
            reveal_path=updates_dir())
    _last = {"hash_failed": False}

    def _validate(path: str) -> bool:
        # Format sanity AND content integrity: verify SHA-256 against GitHub's
        # digest so a corrupted-but-right-size download is rejected and retried
        # rather than installed.  A persistent mismatch fails the download →
        # the UI tells the user to install manually instead of shipping a broken
        # exe (the cause of the "decompression -3" / "Python DLL not found"
        # crashes on networks/AV that mangle the transfer).
        if not _validate_download(path):
            return False
        if not _digest_matches(path, digest):
            _last["hash_failed"] = True
            return False
        _last["hash_failed"] = False
        return True

    try:
        net_download.download_file(
            asset["url"], dest,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            status_cb=status_cb,
            validate_cb=_validate,
            max_attempts=6)
    except net_download.DownloadError as exc:
        if _last["hash_failed"]:
            raise UpdaterError(
                "The download kept failing its integrity check — its SHA-256 "
                "didn't match GitHub's, so the file is being corrupted in "
                "transit (most likely this network or an antivirus product). "
                "Nothing was installed. Download the installer manually from "
                "the Releases page instead.",
                reveal_path=updates_dir()) from exc
        raise UpdaterError(str(exc), reveal_path=updates_dir()) from exc
    return dest


# ── Locating the current install ──────────────────────────────────────────────
def current_app_bundle_path(exe: Optional[str] = None) -> Optional[str]:
    """On macOS, walk up from the executable to the enclosing ``*.app``
    bundle and return it; None if not inside a bundle.  ``exe`` overrides
    ``sys.executable`` (for tests)."""
    exe = exe or sys.executable
    try:
        parts = os.path.realpath(exe).split(os.sep)
    except Exception:
        parts = exe.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(".app"):
            return os.sep.join(parts[: i + 1])
    return None


# ── Helper-script builders (pure → testable) ──────────────────────────────────
def macos_helper_script(pid: int, dmg: str, target_app: str,
                        log_path: str) -> str:
    """Bash helper that waits for ``pid`` to exit, swaps the new
    ``FIREFLY.app`` from ``dmg`` into ``target_app``, clears quarantine,
    relaunches, and self-deletes."""
    # Read pid/dmg/target/log from the POSITIONAL ARGS the Popen call already
    # passes ($1..$4), NOT by interpolating the paths into the script text — a
    # path containing a double-quote, $ or backtick would otherwise break the
    # assignment or expand metacharacters, corrupting the swap (or worse).  (#21)
    return f"""#!/bin/bash
set -u
PID="$1"
DMG="$2"
TARGET="$3"
LOG="$4"
exec >>"$LOG" 2>&1
echo "[firefly-update] waiting for pid $PID to exit"
waited=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 0.3
  waited=$((waited + 1))
  if [ "$waited" -ge 50 ]; then
    echo "[firefly-update] pid $PID still alive after ~15s; force-killing"
    kill -9 "$PID" 2>/dev/null || true
    sleep 1
    break
  fi
done
echo "[firefly-update] mounting $DMG"
MNT="$(mktemp -d /tmp/firefly_upd.XXXXXX)"
if ! hdiutil attach -nobrowse -noverify -noautoopen "$DMG" -mountpoint "$MNT"; then
  echo "[firefly-update] mount failed"
  open -R "$DMG"
  exit 2
fi
NEW="$MNT/FIREFLY.app"
if [ ! -d "$NEW" ]; then
  echo "[firefly-update] FIREFLY.app not found in image"
  hdiutil detach "$MNT" -force
  open -R "$DMG"
  exit 3
fi
BACKUP="${{TARGET}}.bak.$$"
rm -rf "$BACKUP"
echo "[firefly-update] backing up old app -> $BACKUP"
if ! mv "$TARGET" "$BACKUP"; then
  echo "[firefly-update] could not move old app aside"
  hdiutil detach "$MNT" -force
  open -R "$DMG"
  exit 4
fi
echo "[firefly-update] copying new app into place"
if ! cp -R "$NEW" "$TARGET"; then
  echo "[firefly-update] copy failed; restoring backup"
  rm -rf "$TARGET"
  mv "$BACKUP" "$TARGET"
  hdiutil detach "$MNT" -force
  open "$TARGET"
  exit 5
fi
rm -rf "$BACKUP"
echo "[firefly-update] clearing Gatekeeper quarantine"
xattr -dr com.apple.quarantine "$TARGET" 2>/dev/null || true
hdiutil detach "$MNT" -force 2>/dev/null || true
echo "[firefly-update] relaunching"
open "$TARGET"
rm -f "$DMG"
rm -f "$0"
"""


def windows_helper_script(pid: int, new_exe: str, target_exe: str,
                          log_path: str) -> str:
    """Batch helper that waits for ``pid`` to exit, then **safely** swaps
    ``target_exe`` for ``new_exe`` and relaunches.

    Hardened to mirror the macOS path so a failed update can never strand the
    user with a broken install:
      * back up the current exe to ``<target>.bak`` first;
      * copy the new exe in (retrying past Defender's transient file lock);
      * **verify** the copy landed intact — both its byte size AND its SHA-256
        (via ``certutil``) must match the source, so a copy that AV/the
        filesystem corrupted while keeping the size is caught too (the cause of
        the "decompression -3" / "failed to load python3xx.dll" bootloader
        errors);
      * if the copy fails, is short, or its hash differs, **restore the backup**
        so the working version stays in place, and reveal the new exe so the
        user can finish by hand;
      * on success, relaunch with a **ready-marker handshake**.  FIRST, clear
        the inherited PyInstaller one-file markers (`_MEIPASS2` / `_PYI_*`):
        this helper is spawned by the running frozen app, so it inherits the
        bootloader's "I'm the re-exec'd child, skip extraction, use THIS _MEI
        dir" handshake — and if the new exe inherits it, it skips extraction
        and loads python3xx.dll from the OLD (deleted) _MEI dir → "module could
        not be found".  This is THE cause of the recurring DLL error (manual
        double-clicks always worked because Explorer never sets these vars).
        The onefile bundle then re-extracts ~1.4 GB on launch and AV scans the
        fresh exe, so the first extraction can take minutes; the helper clears
        stale _MEI dirs, then waits for the app to signal its window is up for
        as long as the process stays alive (it never kills a still-extracting
        bootloader), and keeps the .bak if no signal arrives within the cap;
      * once the relaunch signals ready (and the copy was already SHA-256
        verified), the new exe is provably good — so the ``.bak`` is **deleted**
        rather than left cluttering the folder.  It is only kept if a check
        fails (restored backup) or the relaunch never signals ready, so a
        manual rollback is still possible exactly when it might be needed.
    """
    # Read pid/exe/target/log from the POSITIONAL ARGS the Popen call already
    # passes (%~1..%~4 — the tilde strips surrounding quotes), NOT by
    # interpolating the paths into the script.  cmd performs %VAR% expansion on
    # a `set "TARGET=<literal-path>"` line, so a path containing '%' (legal in
    # Windows folder names, e.g. C:\50%off\) would be mangled mid-path.  (#32)
    return f"""@echo off
setlocal enabledelayedexpansion
set "PID=%~1"
set "NEWEXE=%~2"
set "TARGET=%~3"
set "LOG=%~4"
set "BACKUP=%TARGET%.bak"
echo [firefly-update] waiting for pid %PID% to exit >>"%LOG%" 2>&1
set /a WAITED=0
:waitloop
tasklist /FI "PID eq %PID%" 2>NUL | find "%PID%" >NUL
if errorlevel 1 goto gone
set /a WAITED+=1
if !WAITED! GEQ 8 (
  echo [firefly-update] pid %PID% still alive after ~16s; force-killing >>"%LOG%" 2>&1
  taskkill /F /T /PID %PID% >>"%LOG%" 2>&1
  ping -n 3 127.0.0.1 >NUL
  goto gone
)
ping -n 2 127.0.0.1 >NUL
goto waitloop
:gone
echo [firefly-update] backing up current exe -^> "%BACKUP%" >>"%LOG%" 2>&1
copy /Y "%TARGET%" "%BACKUP%" >>"%LOG%" 2>&1
set /a TRIES=0
:replace
copy /Y "%NEWEXE%" "%TARGET%" >>"%LOG%" 2>&1
if errorlevel 1 (
  set /a TRIES+=1
  if !TRIES! LSS 20 (
    ping -n 2 127.0.0.1 >NUL
    goto replace
  )
  echo [firefly-update] replace failed after !TRIES! tries; restoring backup >>"%LOG%" 2>&1
  copy /Y "%BACKUP%" "%TARGET%" >>"%LOG%" 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
set "SRCSIZE="
set "DSTSIZE="
for %%A in ("%NEWEXE%") do set "SRCSIZE=%%~zA"
for %%A in ("%TARGET%") do set "DSTSIZE=%%~zA"
if not "!SRCSIZE!"=="!DSTSIZE!" (
  echo [firefly-update] size mismatch src=!SRCSIZE! dst=!DSTSIZE!; restoring backup >>"%LOG%" 2>&1
  copy /Y "%BACKUP%" "%TARGET%" >>"%LOG%" 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
rem ── Content check: the copied exe's SHA-256 must match the (already
rem    GitHub-verified) source.  Catches a copy that AV/the filesystem mangled
rem    while keeping the size — the cause of "decompression -3" at launch.
rem    Best-effort: if certutil is unavailable, fall back to the size check.
set "SRCHASH="
set "DSTHASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%NEWEXE%" SHA256 2^>NUL') do if not defined SRCHASH set "SRCHASH=%%H"
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%TARGET%" SHA256 2^>NUL') do if not defined DSTHASH set "DSTHASH=%%H"
set "SRCHASH=!SRCHASH: =!"
set "DSTHASH=!DSTHASH: =!"
if defined SRCHASH if defined DSTHASH if /i not "!SRCHASH!"=="!DSTHASH!" (
  echo [firefly-update] CONTENT hash mismatch - copy was corrupted; restoring backup >>"%LOG%" 2>&1
  copy /Y "%BACKUP%" "%TARGET%" >>"%LOG%" 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
echo [firefly-update] swap verified (size + SHA-256) >>"%LOG%" 2>&1
rem ── Clear inherited PyInstaller one-file bootloader markers ───────────────
rem THE root cause of the recurring "Failed to load Python DLL python3xx.dll —
rem The specified module could not be found" on relaunch:  this .bat was
rem spawned BY the running frozen FIREFLY, so it inherited _MEIPASS2 / _PYI_*
rem — the onefile bootloader's "I'm the re-exec'd child, skip extraction, use
rem THIS _MEI dir" handshake.  If the new exe inherits them it does NOT extract
rem and tries to load python3xx.dll from the OLD _MEI dir we delete below.
rem Clearing them here (belt-and-suspenders with the cleaned env passed to the
rem helper) forces a clean fresh extraction.  Manual double-clicks always
rem worked precisely because Explorer never sets these.
set "_MEIPASS2="
set "_MEIPASS="
set "_PYI_PARENT_PROCESS_LEVEL="
set "_PYI_ARCHIVE_FILE="
set "_PYI_APPLICATION_HOME_DIR="
set "_PYI_ONEDIR_MODE="
set "_PYI_SPLASH_IPC="
rem ── Relaunch (clear _MEI ONCE before launch, launch ONCE, NEVER kill) ───
rem The new exe is a PyInstaller ONEFILE bundle: every launch re-extracts
rem ~1.4 GB / 10k files to %LOCALAPPDATA%\\FIREFLY\\bundle\\_MEIxxxxx, which
rem Windows Defender scans hard on first run -- it can take SEVERAL MINUTES.
rem TWO past mistakes corrupted that extraction (-> "Failed to load Python DLL
rem python3xx.dll"): (1) force-killing the bootloader mid-extraction, and
rem (2) wiping ALL _MEI* dirs WHILE the new instance was extracting into one.
rem So now we clear stale _MEI dirs EXACTLY ONCE -- AFTER the old app exited
rem and BEFORE launching, so nothing live is touched -- launch ONCE, and NEVER
rem kill the launched process (a slow extraction is healthy).  We wait only to
rem tidy up the .bak after a confirmed-good start; if no ready signal arrives
rem in a generous window we just stop waiting and KEEP the .bak for rollback.
set "BUNDLE=%LOCALAPPDATA%\\FIREFLY\\bundle"
set "MARKER=%~dp0firefly_relaunch_ok.marker"
del "%MARKER%" >NUL 2>&1
set "SPTPALM_READY_MARKER=%MARKER%"
rem Give the just-exited old instance a moment to release its own _MEI handle.
ping -n 4 127.0.0.1 >NUL
rem Clear stale / partially-extracted _MEI dirs ONCE, BEFORE launching.  A dir
rem still locked won't delete and is skipped.  This must NEVER run after launch
rem -- doing so would strip the new instance's python3xx.dll mid-extraction.
for /d %%D in ("%BUNDLE%\\_MEI*") do rmdir /s /q "%%D" >NUL 2>&1
echo [firefly-update] launching new build (onefile extraction can take a few minutes) >>"%LOG%" 2>&1
start "" "%TARGET%"
set /a WAITM=0
:waitmarker
if exist "%MARKER%" goto ready
ping -n 3 127.0.0.1 >NUL
set /a WAITM+=1
rem ~300 x 3s ~= 15 min.  NEVER kill the process -- just stop waiting and keep
rem the .bak so the user can roll back manually if the new build won't start.
if !WAITM! LSS 300 goto waitmarker
echo [firefly-update] no ready signal in ~15 min; kept "%BACKUP%" for manual rollback >>"%LOG%" 2>&1
del "%MARKER%" >NUL 2>&1
goto end
:ready
rem Launched cleanly AND the copy was already SHA-256-verified, so the new exe
rem is provably good -- delete the backup instead of leaving it cluttering the
rem folder (e.g. a visible FIREFLY.exe.bak on the user's Desktop).
del "%MARKER%" >NUL 2>&1
del "%BACKUP%" >NUL 2>&1
echo [firefly-update] update complete + verified; removed backup >>"%LOG%" 2>&1
:end
del "%~f0"
"""


def _write_helper(name: str, content: str, executable: bool) -> str:
    path = os.path.join(updates_dir(), name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(content)
    if executable:
        try:
            os.chmod(path, 0o755)
        except Exception:
            pass
    return path


# ── Apply the update ──────────────────────────────────────────────────────────
def apply_update(downloaded_path: str) -> None:
    """Stage the OS-specific helper and spawn it detached, then return.
    The CALLER must immediately quit the app so the helper can swap files.

    Raises ``UpdaterError`` (with a ``reveal_path`` to the staged file)
    if the update can't be applied automatically — the UI then tells the
    user to finish the install by hand.
    """
    if not is_frozen():
        raise UpdaterError(
            "In-app update only applies to the packaged FIREFLY app.  "
            "Running from source — use 'git pull' instead.")
    if not downloaded_path or not os.path.exists(downloaded_path):
        raise UpdaterError("The downloaded installer is missing.",
                           reveal_path=updates_dir())

    log_path = os.path.join(updates_dir(), "relaunch.log")
    pid = os.getpid()
    devnull = subprocess.DEVNULL

    # CRITICAL — strip the inherited PyInstaller one-file bootloader markers.
    # This helper is spawned BY the running frozen app, so it inherits
    # `_MEIPASS2` (and the `_PYI_*` family) — the onefile bootloader's
    # "I'm the re-exec'd child, DON'T extract, use THIS _MEI dir" handshake.
    # If the relaunched exe inherits them, its bootloader skips extraction and
    # tries to load python3xx.dll from the OLD _MEI dir (which the helper then
    # deletes) → "Failed to load Python DLL … The specified module could not be
    # found".  Passing a cleaned env (and re-clearing inside the .bat) forces a
    # clean, fresh extraction — this is the real cause of the recurring error.
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("_MEIPASS2", "_MEIPASS")
                 and not k.startswith("_PYI")}

    if is_macos():
        target_app = current_app_bundle_path()
        if not target_app or not os.path.isdir(target_app):
            raise UpdaterError(
                "Couldn't locate the FIREFLY.app bundle to update.",
                reveal_path=updates_dir())
        if not os.access(os.path.dirname(target_app), os.W_OK):
            raise UpdaterError(
                f"FIREFLY is installed somewhere that needs administrator "
                f"rights to update ({target_app}).  The new version has "
                f"been downloaded — open it and drag FIREFLY.app over the "
                f"old one to finish.",
                reveal_path=updates_dir())
        helper = _write_helper(
            "firefly_update.sh",
            macos_helper_script(pid, downloaded_path, target_app, log_path),
            executable=True)
        try:
            subprocess.Popen(
                ["/bin/bash", helper, str(pid), downloaded_path,
                 target_app, log_path],
                start_new_session=True, close_fds=True, env=clean_env,
                stdin=devnull, stdout=devnull, stderr=devnull)
        except Exception as exc:
            raise UpdaterError(f"Couldn't start the updater: {exc}",
                               reveal_path=updates_dir()) from exc
        return

    if is_windows():
        target_exe = sys.executable
        if not os.access(os.path.dirname(target_exe) or ".", os.W_OK):
            raise UpdaterError(
                f"FIREFLY is installed somewhere that needs administrator "
                f"rights to update ({target_exe}).  The new version has "
                f"been downloaded — run it to finish updating.",
                reveal_path=updates_dir())
        helper = _write_helper(
            "firefly_update.bat",
            windows_helper_script(pid, downloaded_path, target_exe, log_path),
            executable=False)
        # NB: CREATE_NO_WINDOW must NOT be combined with DETACHED_PROCESS —
        # Windows ignores CREATE_NO_WINDOW in that combination (per the
        # CreateProcess docs), which pops a visible console.  CREATE_NO_WINDOW
        # alone gives a hidden console; the child still outlives us (Windows
        # doesn't kill orphans), so no DETACHED_PROCESS is needed.
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        try:
            subprocess.Popen(
                ["cmd", "/c", helper, str(pid), downloaded_path,
                 target_exe, log_path],
                creationflags=(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP),
                close_fds=True, env=clean_env,
                stdin=devnull, stdout=devnull, stderr=devnull)
        except Exception as exc:
            raise UpdaterError(f"Couldn't start the updater: {exc}",
                               reveal_path=updates_dir()) from exc
        return

    raise UpdaterError("Automatic update isn't supported on this platform.",
                       reveal_path=updates_dir())
