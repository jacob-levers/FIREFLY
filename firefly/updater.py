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
    """GET the GitHub 'latest release' JSON.  Returns the parsed dict, or
    an empty dict on ANY failure (offline, rate-limited, bad JSON) so
    callers can treat "no update" and "couldn't check" uniformly."""
    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "FIREFLY-app"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read()
        data = json.loads(blob)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def select_asset(release_json: dict,
                 asset_name: Optional[str]) -> Optional[dict]:
    """Find the named asset in a release JSON; return
    ``{"name", "url", "size"}`` or None if absent."""
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
                    "size": int(a.get("size") or 0)}
    return None


def parse_release(release_json: dict) -> dict:
    """Normalise a GitHub release JSON into the fields the UI needs:
    ``{"tag", "html_url", "body", "asset"}``.  ``asset`` is the installer
    for the current OS (or None if this platform has no matching asset)."""
    if not isinstance(release_json, dict):
        release_json = {}
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
    try:
        net_download.download_file(
            asset["url"], dest,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            status_cb=status_cb,
            validate_cb=_validate_download,
            max_attempts=6)
    except net_download.DownloadError as exc:
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
    return f"""#!/bin/bash
set -u
PID="{pid}"
DMG="{dmg}"
TARGET="{target_app}"
LOG="{log_path}"
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
      * **verify** the copy landed intact — its byte size must match the
        source (a truncated copy is the classic cause of the "failed to load
        python3xx.dll" bootloader error);
      * if the copy fails or is short, **restore the backup** so the working
        version stays in place, and reveal the new exe so the user can finish
        by hand;
      * on success, relaunch and KEEP the ``.bak`` so the user can roll back
        manually if the new build won't start on their machine.
    """
    return f"""@echo off
setlocal enabledelayedexpansion
set "PID={pid}"
set "NEWEXE={new_exe}"
set "TARGET={target_exe}"
set "LOG={log_path}"
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
echo [firefly-update] swap verified (size !DSTSIZE!); relaunching >>"%LOG%" 2>&1
start "" "%TARGET%"
echo [firefly-update] previous version kept at "%BACKUP%" (rename to roll back) >>"%LOG%" 2>&1
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
                start_new_session=True, close_fds=True,
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
                close_fds=True,
                stdin=devnull, stdout=devnull, stderr=devnull)
        except Exception as exc:
            raise UpdaterError(f"Couldn't start the updater: {exc}",
                               reveal_path=updates_dir()) from exc
        return

    raise UpdaterError("Automatic update isn't supported on this platform.",
                       reveal_path=updates_dir())
