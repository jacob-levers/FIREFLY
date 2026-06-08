"""
FIREFLY general-purpose file downloader.

A small, robust HTTPS downloader shared across the app — the in-app
updater (``firefly/updater.py``) and the CUDA sidecar installer
(``firefly/cuda_installer.py``) both route their plain-urllib downloads
through here so the hard-won robustness lives in one place.  Pure stdlib
(no PySide6), so it is safe to import anywhere, including before the Qt
event loop exists.

Robustness (carried over from cuda_installer's hardened wheel download):
  * Atomic write — bytes land in ``<dest>.part`` and are renamed to the
    final path only after the size + format checks pass, so a partial
    file never looks complete.
  * Resumable — a leftover ``<dest>.part`` from a crashed run is resumed
    with a ``Range:`` header instead of starting over.
  * Stall watchdog — a daemon thread tears the response down if no bytes
    arrive for ``read_stall_s`` seconds, so a wedged TLS connection fails
    fast with a clear error instead of hanging the app forever.
  * Throttled progress — progress callbacks are capped at ~10 Hz so a
    fast connection can't flood the Qt event queue and trip the OS
    "Not Responding" detector.
  * Retry with backoff — transient failures (URLError, timeout, stall,
    short read) re-enter the loop; permanent ones (HTTP 4xx, user
    cancel) abort immediately.
  * Caller-supplied ``validate_cb`` — e.g. ``zipfile.is_zipfile`` for a
    wheel, or a "looks like a DMG / Windows PE" check for an installer.

TLS/CA note: the app installs a process-global certifi SSL context at
startup (see ``firefly/ui/app_qt.py``), so plain
``urllib.request.urlopen`` already verifies certificates in frozen
builds.  We deliberately do NOT re-wire SSL per call here.
"""
from __future__ import annotations

import os
import sys
import time
import threading
import urllib.error
import urllib.request
from typing import Callable, Optional


# ── Diagnostic log plumbing ───────────────────────────────────────────────────
# A consumer (e.g. the update dialog or the CUDA installer dialog) registers a
# callback via set_log_callback(); every _log() line is forwarded there as well
# as printed to stdout, so the user gets a step-by-step breadcrumb trail.
_log_cb: Optional[Callable[[str], None]] = None
_log_t0: float = 0.0


def set_log_callback(cb: Optional[Callable[[str], None]]) -> None:
    """Register a callable that receives each diagnostic line (also
    printed to stdout).  Pass None to clear."""
    global _log_cb, _log_t0
    _log_cb = cb
    _log_t0 = time.monotonic()


def _log(msg: str) -> None:
    """Emit a timestamped diagnostic line."""
    elapsed = time.monotonic() - _log_t0 if _log_t0 else 0.0
    line = f"[+{elapsed:5.2f}s] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    if _log_cb is not None:
        try:
            _log_cb(line)
        except Exception:
            pass


def is_windows() -> bool:
    return sys.platform == "win32"


class DownloadError(RuntimeError):
    """Raised when a download ultimately fails — all retries exhausted, a
    permanent HTTP error (4xx), validation failure, or user cancel."""


def download_file(url: str,
                  dest_path: str,
                  *,
                  progress_cb: Optional[Callable[[int, int], None]] = None,
                  cancel_cb: Optional[Callable[[], bool]] = None,
                  validate_cb: Optional[Callable[[str], bool]] = None,
                  headers: Optional[dict] = None,
                  max_attempts: int = 3,
                  timeout: float = 20.0,
                  read_stall_s: float = 10.0) -> None:
    """Download ``url`` → ``dest_path`` with progress, cancel, resume,
    validation and retry/backoff.

    Args:
      progress_cb(downloaded, total): called (throttled to ~10 Hz) with
        byte counts; ``total`` is 0 if the server sent no Content-Length.
      cancel_cb() -> bool: polled between chunks; return True to abort.
      validate_cb(path) -> bool: called on the finished ``.part`` before
        the atomic rename; return False to reject (treated as a corrupt
        download and retried).
      headers: extra request headers (merged over a default User-Agent).
      max_attempts: total tries before giving up (backoff 2/5/10 s).

    Raises ``DownloadError`` on failure; returns None on success (the
    complete file is at ``dest_path``).
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    # Retry loop.  Backoff (seconds): 2, 5, 10 — rides out a transient
    # firewall hiccup without boring the user.  The .part file is kept
    # between attempts so a retry resumes rather than re-downloading.
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            _download_once(url, dest_path,
                           progress_cb=progress_cb, cancel_cb=cancel_cb,
                           validate_cb=validate_cb, headers=headers,
                           attempt=attempt, max_attempts=max_attempts,
                           timeout=timeout, read_stall_s=read_stall_s)
            return
        except DownloadError as exc:
            # User-cancel + permanent server errors (4xx) propagate
            # immediately — don't burn retries on a 404 or a cancel.
            msg = str(exc).lower()
            if "cancel" in msg or "http 4" in msg:
                raise
            last_exc = exc
            _log(f"  attempt {attempt}/{max_attempts} failed: {exc}")
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError) as exc:
            last_exc = exc
            _log(f"  attempt {attempt}/{max_attempts} failed "
                 f"({type(exc).__name__}: {exc})")
        if attempt < max_attempts:
            backoff = (2, 5, 10)[min(attempt - 1, 2)]
            _log(f"  backing off {backoff}s before retry "
                 f"(resume buffer kept)")
            for _ in range(backoff):
                if cancel_cb is not None:
                    try:
                        if cancel_cb():
                            raise DownloadError("Download cancelled by user.")
                    except DownloadError:
                        raise
                    except Exception:
                        pass
                time.sleep(1)

    # All attempts exhausted — clean up the .part and report failure.
    part_path = dest_path + ".part"
    try:
        if os.path.exists(part_path):
            os.remove(part_path)
    except Exception:
        pass
    raise DownloadError(
        f"Download failed after {max_attempts} attempts.  "
        f"Last error: {last_exc}")


def _download_once(url: str,
                   dest_path: str,
                   *,
                   progress_cb: Optional[Callable[[int, int], None]],
                   cancel_cb: Optional[Callable[[], bool]],
                   validate_cb: Optional[Callable[[str], bool]],
                   headers: Optional[dict],
                   attempt: int,
                   max_attempts: int,
                   timeout: float,
                   read_stall_s: float) -> None:
    """One attempt: write to ``<dest>.part`` (resuming any prior one),
    verify, then atomic-rename to ``dest``.  Raises on any failure —
    ``download_file`` decides whether to retry."""
    part_path = dest_path + ".part"
    resume_from = 0
    try:
        if os.path.exists(part_path):
            resume_from = int(os.path.getsize(part_path))
    except Exception:
        resume_from = 0

    # 256 KB chunks + a 10 Hz progress cap keep the Qt event queue from
    # being flooded on a fast connection (which makes the OS mark the
    # app "Not Responding").
    chunk_size = 256 * 1024
    progress_throttle_s = 0.1
    last_progress_t = 0.0

    # Always re-derive the FINAL file from .part; leave .part as the
    # resume buffer.
    try:
        if os.path.exists(dest_path):
            os.remove(dest_path)
    except Exception:
        pass

    _log(f"GET {url}  (attempt {attempt}/{max_attempts}"
         + (f", resume from {resume_from/1e6:.1f} MB" if resume_from else "")
         + ")")

    req_headers = {"User-Agent": "FIREFLY-app"}
    if headers:
        req_headers.update(headers)
    if resume_from > 0:
        req_headers["Range"] = f"bytes={resume_from}-"

    # Watchdog: resp.read() can block forever on a stalled TLS stream.
    # Sample byte progress every 1 s in a daemon thread; if nothing
    # arrives for read_stall_s, close the response so the worker's read
    # returns/raises and we fail with a clear error instead of hanging.
    progress_state = {"downloaded": 0, "last_change_t": time.monotonic(),
                      "should_abort": False, "done": False}
    resp_holder: dict = {"resp": None}

    def _stall_watchdog():
        while not progress_state["should_abort"]:
            time.sleep(1.0)
            if progress_state.get("done"):
                return
            elapsed = time.monotonic() - progress_state["last_change_t"]
            if elapsed > read_stall_s:
                _log(f"  → STALL WATCHDOG: no data for {elapsed:.0f}s, "
                     f"aborting (downloaded "
                     f"{progress_state['downloaded']/1e6:.1f} MB)")
                progress_state["should_abort"] = True
                try:
                    r = resp_holder.get("resp")
                    if r is not None:
                        r.close()
                except Exception:
                    pass
                return

    try:
        req = urllib.request.Request(url, headers=req_headers)
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_holder["resp"] = resp
            status = int(getattr(resp, "status", 0) or 0)
            _log(f"  HTTP {status} in {time.monotonic()-t0:.2f}s")
            # Asked for a range but got 200 → server ignored it; discard
            # the .part and start fresh rather than corrupt the file.
            if resume_from > 0 and status != 206:
                _log(f"  server returned {status} (not 206 Partial Content) "
                     f"— discarding {resume_from/1e6:.1f} MB resume buffer")
                resume_from = 0
                try: os.remove(part_path)
                except Exception: pass
            try:
                cl = int(resp.headers.get("Content-Length") or 0)
            except Exception:
                cl = 0
            # On a 206, Content-Length is the REMAINDER, not the total.
            total = (resume_from + cl) if (status == 206 and cl > 0) else cl
            wdog = threading.Thread(target=_stall_watchdog, daemon=True,
                                    name="firefly-download-stall-watchdog")
            wdog.start()
            downloaded = resume_from
            progress_state["downloaded"] = downloaded
            file_mode = "ab" if resume_from > 0 else "wb"
            with open(part_path, file_mode) as out:
                while True:
                    if cancel_cb is not None:
                        try:
                            if cancel_cb():
                                # Leave .part for resume on re-run.
                                progress_state["should_abort"] = True
                                raise DownloadError(
                                    "Download cancelled by user.")
                        except DownloadError:
                            raise
                        except Exception:
                            pass
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    if progress_state["should_abort"]:
                        raise DownloadError(
                            f"Download stalled — no data received for "
                            f"{read_stall_s:.0f} seconds.")
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    progress_state["downloaded"] = downloaded
                    progress_state["last_change_t"] = now
                    if (progress_cb is not None
                            and (now - last_progress_t) >= progress_throttle_s):
                        last_progress_t = now
                        try:
                            progress_cb(downloaded, total)
                        except Exception:
                            pass
            progress_state["done"] = True
            _log(f"  ✓ download complete: {downloaded/1e6:.1f} MB in "
                 f"{time.monotonic()-t0:.1f}s")
            if progress_cb is not None:
                try:
                    progress_cb(downloaded, total)
                except Exception:
                    pass

        # ── Integrity gauntlet ───────────────────────────────────────
        # 1) Size matches Content-Length (if the server sent one).
        try:
            actual = os.path.getsize(part_path)
        except Exception:
            actual = 0
        if total > 0 and actual != total:
            raise DownloadError(
                f"Short read: got {actual/1e6:.1f} MB but Content-Length "
                f"indicated {total/1e6:.1f} MB.")
        # 2) Caller-supplied format check (catches a captive-portal HTML
        #    page that slipped through as a 200).
        if validate_cb is not None:
            try:
                ok = bool(validate_cb(part_path))
            except Exception:
                ok = False
            if not ok:
                try: os.remove(part_path)
                except Exception: pass
                raise DownloadError(
                    "Downloaded file failed validation (unexpected format "
                    "— an intercepting proxy may have returned an error "
                    "page).  Try again on a different network.")
        # 3) Atomic rename — only if every earlier check passed.
        try:
            os.replace(part_path, dest_path)
        except OSError as exc:
            # Most common on Windows: Defender has the .part open for
            # scanning.  Wait a beat and retry once.
            _log(f"  rename failed ({exc}); retrying after 1s (Defender scan?)")
            time.sleep(1.0)
            os.replace(part_path, dest_path)
    except urllib.error.HTTPError as exc:
        progress_state["should_abort"] = True
        code = getattr(exc, "code", 0)
        # 416 = Range Not Satisfiable — our .part is bigger than the
        # server's file now (asset re-uploaded between attempts).  Discard.
        if code == 416:
            _log("  HTTP 416 — discarding stale .part and retrying fresh")
            try: os.remove(part_path)
            except Exception: pass
            raise urllib.error.URLError(f"416: {exc.reason}") from exc
        # 4xx → permanent.  Tag "HTTP 4" so download_file aborts instead
        # of burning retries on a 404.
        if 400 <= code < 500:
            raise DownloadError(
                f"HTTP {code} when downloading {url}") from exc
        # 5xx → transient; let the retry loop have it.
        raise urllib.error.URLError(f"{code}: {exc.reason}") from exc
    except urllib.error.URLError:
        progress_state["should_abort"] = True
        raise           # outer retry loop catches it; keep .part for resume
    except DownloadError:
        progress_state["should_abort"] = True
        raise           # cancel / short-read / stall / validation
    except Exception as exc:
        progress_state["should_abort"] = True
        raise DownloadError(
            f"Unexpected error while downloading: {exc}") from exc
    finally:
        progress_state["should_abort"] = True
