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
The Qt layer spawns that helper, then quits cleanly so the running application
releases its files before the installer swaps them.

macOS note: the app is unsigned/un-notarised, so the helper clears the
Gatekeeper quarantine flag (``xattr -dr com.apple.quarantine``) on the
freshly-copied bundle, mirroring the manual "right-click → Open" step.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
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


def current_os_asset_names() -> tuple[str, ...]:
    """Ordered release-asset names compatible with this exact platform.

    macOS releases are Apple-Silicon builds with an architecture-explicit
    artifact name.  A generic DMG is intentionally not accepted: it cannot
    communicate whether the binary is compatible with the current Mac.
    """
    if is_macos():
        machine = platform.machine().lower()
        if machine in {"arm64", "aarch64"}:
            return ("FIREFLY-macOS-arm64.dmg",)
        return ()
    if is_windows():
        return ("FIREFLY-Windows.exe",)
    return ()


def current_os_asset_name() -> Optional[str]:
    """Primary release asset for this platform, or ``None`` if unsupported."""
    names = current_os_asset_names()
    if names:
        return names[0]
    return None


def installer_unavailable_message() -> str:
    """Actionable platform/asset error for the updater UI."""
    if is_macos():
        machine = platform.machine().lower()
        if machine not in {"arm64", "aarch64"}:
            return (
                "FIREFLY's pre-built macOS app supports Apple Silicon (arm64) "
                "only; no Intel macOS installer is published. Use the source "
                "installation instructions instead.")
        return (
            "This release is missing FIREFLY-macOS-arm64.dmg. "
            "Open Release notes to download a published installer manually.")
    if is_windows():
        return (
            "This release is missing FIREFLY-Windows.exe. "
            "Open Release notes to download a published installer manually.")
    return "No FIREFLY installer is published for this platform."


def updates_dir() -> str:
    """`<app-data>/FIREFLY/updates`, created on first use.  Staging area
    for the downloaded installer + the helper script + relaunch log."""
    d = os.path.join(os.path.dirname(crash_reporter.crash_report_dir()),
                     "updates")
    os.makedirs(d, exist_ok=True)
    return d


# ── Version comparison ────────────────────────────────────────────────────────
def parse_version(s: str) -> "tuple[int, ...]":
    """Parse a 'v2.41.0' / '2.41.0-rc.2' style tag into a comparable tuple.

    Pre-release aware (semver ordering): a suffix sorts BEFORE the final release
    of the same x.y.z, and numbered pre-releases order among themselves —
    ``2.76.39-rc.1 < 2.76.39-rc.2 < 2.76.39``.  Encoded as
    ``(major, minor, patch, is_final, pre_num)``: a final release is
    ``(…, 1, 0)``; a pre-release is ``(…, 0, n)`` where ``n`` is the trailing
    integer in the suffix (``rc.2`` → 2, none → 0).  The single canonical
    comparator — ``is_newer`` / ``pick_release`` delegate here."""
    import re
    raw = (s or "").lstrip("vV")
    core, _, pre = raw.partition("-")
    parts = []
    for chunk in core.split("."):
        m = re.match(r"(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    parts = (parts + [0, 0, 0])[:3]
    if pre:
        m = re.search(r"(\d+)\s*$", pre)
        return tuple(parts) + (0, int(m.group(1)) if m else 0)
    return tuple(parts) + (1, 0)


def is_newer(latest: str, current: str) -> bool:
    """True if release tag ``latest`` is strictly newer than ``current``."""
    return parse_version(latest) > parse_version(current)


def is_prerelease_version(s: str) -> bool:
    """True if a version string carries a pre-release suffix (e.g. ``2.76.39-rc.1``).
    Used to offer a return to the stable release when a beta build is running on
    the Stable channel."""
    return "-" in (s or "").lstrip("vV")


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


def fetch_releases(api_url: str, timeout: float = 6.0) -> "list | dict":
    """GET the GitHub *releases list* (newest-first), used to resolve the update
    channel.  Returns a list of release-JSON dicts, ``[]`` on any failure, or the
    ``{"_rate_limited": True, "_reset": <epoch>}`` marker on a rate limit (same
    semantics as :func:`fetch_latest_release`).

    The ``/releases/latest`` endpoint only ever returns the newest NON-prerelease
    by GitHub's definition, so it can't see betas — the Pre-release channel and
    "notify about pre-releases" need this list endpoint instead."""
    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "FIREFLY-app"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read()
        data = json.loads(blob)
        return data if isinstance(data, list) else []
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
        return []
    except Exception:
        return []


def pick_release(releases: "list",
                 include_prerelease: bool) -> Optional[dict]:
    """From a GitHub releases list, return the newest installable release JSON
    (highest version tag), skipping drafts and — unless ``include_prerelease`` —
    prereleases.  ``None`` if the list is empty or nothing qualifies.

    Chooses by parsed version, not list order, so an out-of-order or
    re-published release can't pick a stale "latest"."""
    if not isinstance(releases, list):
        return None
    best = None
    best_ver = None
    for r in releases:
        if not isinstance(r, dict):
            continue
        if r.get("draft"):
            continue
        if r.get("prerelease") and not include_prerelease:
            continue
        tag = r.get("tag_name") or ""
        if not tag:
            continue
        ver = parse_version(tag)
        if best_ver is None or ver > best_ver:
            best_ver, best = ver, r
    return best


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
    asset = None
    for name in current_os_asset_names():
        asset = select_asset(release_json, name)
        if asset is not None:
            break
    return {
        "tag": release_json.get("tag_name") or "",
        "html_url": release_json.get("html_url") or "",
        "body": release_json.get("body") or "",
        "asset": asset,
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
    """Cheap, deterministic sanity check for FIREFLY's UDIF disk image.

    A UDIF image ends with a 512-byte ``koly`` trailer.  Checking that local
    structure rejects an HTML error page or an obviously truncated response
    without spawning ``hdiutil imageinfo`` after progress has reached 100%.
    That subprocess was both slow and fallible; a transient local failure was
    reported as corrupt network bytes and caused a complete second download.

    This is intentionally only a format gate.  :func:`download_asset` separately
    authenticates every byte against GitHub's published SHA-256 before this
    result is accepted, and the detached helper still asks ``hdiutil`` to mount
    the image during installation.
    """
    try:
        if os.path.getsize(path) < 1_000_000:
            return False
        with open(path, "rb") as fh:
            fh.seek(-512, os.SEEK_END)
            return fh.read(4) == b"koly"
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


# Auto-download and a click on "Download & install" can arrive on separate
# threads.  Both historically wrote the same ``<asset>.part`` and ``.segN``
# files, corrupting one another's resume/progress state.  Serialize per final
# destination and let later callers reuse the first authenticated result.
_asset_locks_guard = threading.Lock()
_asset_locks: dict[str, threading.Lock] = {}


def _asset_lock(dest: str) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(dest))
    with _asset_locks_guard:
        lock = _asset_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _asset_locks[key] = lock
        return lock


def _asset_identity_path(dest: str) -> str:
    return dest + ".source.json"


def _asset_identity(asset: dict) -> dict:
    """Stable identity for resumable bytes belonging to one release asset."""
    return {
        "url": str(asset.get("url") or ""),
        "size": int(asset.get("size") or 0),
        "digest": str(asset.get("digest") or "").strip().lower(),
    }


def _prepare_resume_state(dest: str, identity: dict) -> None:
    """Keep resume files only when their sidecar identifies this exact asset.

    GitHub uses the same installer filename for every FIREFLY release.  Without
    an identity sidecar, a partial from an older version was sent as the prefix
    of the new version and failed SHA-256 only after the bar reached 100%; the
    subsequent retry then visibly downloaded the installer from zero.
    """
    sidecar = _asset_identity_path(dest)
    previous = None
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            value = json.load(fh)
        if isinstance(value, dict):
            previous = value
    except Exception:
        previous = None

    if previous != identity:
        # The contiguous ``.part`` is the only state the updater will resume
        # (release downloads deliberately disable segmentation below).  Its
        # removal is mandatory: never write the new identity if AV/a file lock
        # leaves old-release prefix bytes behind.
        part = dest + ".part"
        delay = 0.1
        last_exc = None
        for attempt in range(6):
            if not os.path.exists(part):
                break
            try:
                os.remove(part)
                break
            except OSError as exc:
                last_exc = exc
                if attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 1.0)
        if os.path.exists(part):
            raise UpdaterError(
                "Couldn't discard resume data from an older FIREFLY release "
                f"({last_exc}). Close antivirus/file-indexing tools or install "
                "the update manually from the Releases page.",
                reveal_path=updates_dir())

        # Old final/segmented files are never resumed by this code path.  Clean
        # them best-effort; a persistent lock will be diagnosed by the normal
        # final replace rather than allowing stale bytes into the new stream.
        stale = [dest]
        stale.extend(f"{dest}.part.seg{i}" for i in range(64))
        for path in stale:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    # Write atomically so a process killed here cannot bless unidentified bytes
    # as belonging to the next run.
    tmp = sidecar + f".{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(identity, fh, sort_keys=True)
        os.replace(tmp, sidecar)
    except Exception as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise UpdaterError(
            f"Couldn't prepare the update staging area: {exc}",
            reveal_path=updates_dir()) from exc


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
        raise UpdaterError(installer_unavailable_message())
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
    expected_size = int(asset.get("size") or 0)

    def _validate(path: str) -> bool:
        # Format sanity AND content integrity: verify SHA-256 against GitHub's
        # digest so a corrupted-but-right-size download is rejected and retried
        # rather than installed.  A persistent mismatch fails the download →
        # the UI tells the user to install manually instead of shipping a broken
        # exe (the cause of the "decompression -3" / "Python DLL not found"
        # crashes on networks/AV that mangle the transfer).
        if expected_size:
            try:
                if os.path.getsize(path) != expected_size:
                    return False
            except Exception:
                return False
        if not _validate_download(path):
            return False
        if not _digest_matches(path, digest):
            _last["hash_failed"] = True
            return False
        _last["hash_failed"] = False
        return True

    lock = _asset_lock(dest)
    announced_wait = False
    while not lock.acquire(timeout=0.1):
        if not announced_wait and status_cb is not None:
            try:
                status_cb("Waiting for the background download…")
            except Exception:
                pass
            announced_wait = True
        if cancel_cb is not None:
            try:
                if cancel_cb():
                    raise UpdaterError("Download cancelled by user.")
            except UpdaterError:
                raise
            except Exception:
                pass

    try:
        # Covers both an overlapping background prefetch and a second click
        # after helper staging failed: never fetch an installer we already
        # authenticated for this release.
        if os.path.isfile(dest) and _validate(dest):
            if progress_cb is not None:
                try:
                    size = os.path.getsize(dest)
                    progress_cb(size, expected_size or size)
                except Exception:
                    pass
            # Emit the non-transfer phase last so a watching controller does
            # not remain at the synthetic 100% cache-hit report.
            if status_cb is not None:
                try:
                    status_cb("Using verified download…")
                except Exception:
                    pass
            return dest

        # A stale complete installer may have failed the cache fast-path hash.
        # Do not let that result mislabel an unrelated network failure below as
        # repeated in-transit corruption.
        _last["hash_failed"] = False
        _prepare_resume_state(dest, _asset_identity(asset))
        try:
            net_download.download_file(
                asset["url"], dest,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                status_cb=status_cb,
                validate_cb=_validate,
                max_attempts=6,
                # Release installers use one coherent, resumable stream.  A
                # late segmented/proxy failure otherwise discards discontiguous
                # ranges and visibly starts an entire second transfer.
                parallel_segments=1)
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
    finally:
        lock.release()


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

    No backup of the old exe is kept.  Instead of copying the new build straight
    over the target (which, if it failed or was AV-mangled mid-copy, would strand
    a broken install and needed a ``.bak`` to restore), it **stages then swaps**:
      * copy the new exe to ``<target>.new`` (retrying past Defender's transient
        file lock);
      * **verify** the staged copy — both its byte size AND its SHA-256 (via
        ``certutil``) must match the source, catching a copy that AV/the
        filesystem corrupted while keeping the size (the cause of the
        "decompression -3" / "failed to load python3xx.dll" bootloader errors);
      * only then **rename** ``<target>.new`` over the target — a fast, same-
        folder move, so the working old build stays untouched until that final
        step.  If the copy, verify, or rename fails, the target is left as the
        (still-working) old build and the new exe is revealed so the user can
        finish by hand — no backup required;
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
        bootloader).
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
set "STAGED=%TARGET%.new"
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
rem ── Stage the new exe next to the target, verify it, then swap by a fast
rem    same-folder rename.  No backup of the old exe is kept: the target is only
rem    touched by a rename of an ALREADY-verified file, so a failed or AV-mangled
rem    copy leaves the working old build in place (until the final rename).
del "%STAGED%" >NUL 2>&1
set /a TRIES=0
:stage
copy /Y "%NEWEXE%" "%STAGED%" >>"%LOG%" 2>&1
if errorlevel 1 (
  set /a TRIES+=1
  if !TRIES! LSS 20 (
    ping -n 2 127.0.0.1 >NUL
    goto stage
  )
  echo [firefly-update] staging copy failed after !TRIES! tries; target untouched >>"%LOG%" 2>&1
  del "%STAGED%" >NUL 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
set "SRCSIZE="
set "DSTSIZE="
for %%A in ("%NEWEXE%") do set "SRCSIZE=%%~zA"
for %%A in ("%STAGED%") do set "DSTSIZE=%%~zA"
if not "!SRCSIZE!"=="!DSTSIZE!" (
  echo [firefly-update] staged size mismatch src=!SRCSIZE! dst=!DSTSIZE!; target untouched >>"%LOG%" 2>&1
  del "%STAGED%" >NUL 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
rem ── Content check: the staged exe's SHA-256 must match the (already
rem    GitHub-verified) source.  Catches a copy that AV/the filesystem mangled
rem    while keeping the size — the cause of "decompression -3" at launch.
rem    Best-effort: if certutil is unavailable, fall back to the size check.
set "SRCHASH="
set "DSTHASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%NEWEXE%" SHA256 2^>NUL') do if not defined SRCHASH set "SRCHASH=%%H"
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%STAGED%" SHA256 2^>NUL') do if not defined DSTHASH set "DSTHASH=%%H"
set "SRCHASH=!SRCHASH: =!"
set "DSTHASH=!DSTHASH: =!"
if defined SRCHASH if defined DSTHASH if /i not "!SRCHASH!"=="!DSTHASH!" (
  echo [firefly-update] staged content hash mismatch - copy corrupted; target untouched >>"%LOG%" 2>&1
  del "%STAGED%" >NUL 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
echo [firefly-update] staged copy verified (size + SHA-256) >>"%LOG%" 2>&1
rem ── Swap by rename (fast, same folder); retry past a transient lock ────────
set /a TRIES=0
:swap
move /Y "%STAGED%" "%TARGET%" >>"%LOG%" 2>&1
if errorlevel 1 (
  set /a TRIES+=1
  if !TRIES! LSS 20 (
    ping -n 2 127.0.0.1 >NUL
    goto swap
  )
  echo [firefly-update] swap rename failed after !TRIES! tries; target untouched >>"%LOG%" 2>&1
  del "%STAGED%" >NUL 2>&1
  start "" explorer.exe /select,"%NEWEXE%"
  goto end
)
echo [firefly-update] swap complete (size + SHA-256 verified) >>"%LOG%" 2>&1
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
rem confirm a good start; if no ready signal arrives in a generous window we
rem just stop waiting (the new build was SHA-256-verified before the swap).
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
rem ~300 x 3s ~= 15 min.  NEVER kill the process -- just stop waiting.  The new
rem build was SHA-256-verified before the swap, so a slow first extraction is
rem normal and there is no backup to clean up.
if !WAITM! LSS 300 goto waitmarker
echo [firefly-update] no ready signal in ~15 min (build was verified before swap) >>"%LOG%" 2>&1
del "%MARKER%" >NUL 2>&1
goto end
:ready
del "%MARKER%" >NUL 2>&1
echo [firefly-update] update complete + verified >>"%LOG%" 2>&1
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
