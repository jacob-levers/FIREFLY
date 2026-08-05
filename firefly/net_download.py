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
    permanent HTTP error (4xx), validation failure, or user cancel.

    ``terminal=True`` marks a failure that re-downloading cannot fix (e.g. the
    OS denied the final rename into place), so ``download_file`` stops instead of
    burning its whole retry budget re-fetching the file for nothing."""

    def __init__(self, *args, terminal: bool = False):
        super().__init__(*args)
        self.terminal = terminal


def _finalize_download(part_path: str, dest_path: str, *, tries: int = 8) -> None:
    """Move the completed ``.part`` onto the final path, retrying past a
    transient Windows file lock.

    On Windows, a freshly-written ``.exe`` (or a leftover previous one already at
    the destination) is routinely held open for a beat by Defender's on-write
    scan, so ``os.replace`` fails with ``[WinError 5] Access is denied``.  A scan
    of a large installer can take several seconds, so retry with backoff (~0.5 →
    8 s, ~30 s total) to ride it out.  If it STILL fails, the cause is a
    persistent lock or a security policy (e.g. AppLocker blocking ``.exe`` in
    AppData) that re-downloading won't fix → raise a *terminal* error so the
    caller fails fast and tells the user to install by hand."""
    delay = 0.5
    last = None
    for i in range(1, tries + 1):
        try:
            os.replace(part_path, dest_path)
            return
        except OSError as exc:
            last = exc
            _log(f"  finalize attempt {i}/{tries} failed ({exc})"
                 + (f"; retry in {delay:.1f}s (Windows/Defender file lock?)"
                    if i < tries else ""))
            if i >= tries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 8.0)
    raise DownloadError(
        f"Couldn't save the update to disk — the operating system denied "
        f"access to the file after {tries} tries ({last}).  This is almost "
        f"always antivirus or a security policy (e.g. AppLocker) blocking the "
        f"installer; nothing was changed.  Download the installer manually from "
        f"the Releases page instead.",
        terminal=True)


def _recover_complete_part(dest_path: str,
                           expected_size: Optional[int],
                           *,
                           progress_cb: Optional[Callable[[int, int], None]],
                           cancel_cb: Optional[Callable[[], bool]],
                           validate_cb: Optional[Callable[[str], bool]],
                           status_cb: Optional[Callable[[str], None]]) -> bool:
    """Validate and finalize an already-complete resume buffer locally.

    A process can be interrupted after writing the last byte but before the
    validation/rename phase.  Sending ``Range: bytes=<size>-`` on the next run
    is an unsatisfiable EOF range, so a compliant server responds with 416 and
    the old implementation discarded and downloaded the complete file again.

    ``expected_size`` must be authoritative for the same asset (the updater,
    for example, gets it from GitHub's release metadata and separately keys its
    resume sidecar by URL/size/digest).  Merely matching that size is never used
    to bypass ``validate_cb``: the local bytes pass the exact same caller-owned
    authentication/format check as freshly downloaded bytes.

    Returns True only when the part was finalized.  A right-sized part that
    fails validation is removed and returns False so the caller downloads a
    clean copy.  Short parts are untouched for ordinary Range resume.
    """
    if expected_size is None or expected_size <= 0:
        return False
    part_path = dest_path + ".part"
    try:
        part_size = os.path.getsize(part_path)
    except OSError:
        return False
    if part_size != expected_size:
        return False

    if cancel_cb is not None:
        try:
            if cancel_cb():
                raise DownloadError("Download cancelled by user.")
        except DownloadError:
            raise
        except Exception:
            pass

    _log(f"  complete resume buffer found ({part_size/1e6:.1f} MB); "
         "verifying locally")
    # Match the end-of-transfer callback order: report all bytes first, then
    # switch the UI into its named, indeterminate verification phase.
    if progress_cb is not None:
        try:
            progress_cb(part_size, expected_size)
        except Exception:
            pass
    if status_cb is not None:
        try:
            status_cb("Verifying download…")
        except Exception:
            pass

    ok = True
    if validate_cb is not None:
        try:
            ok = bool(validate_cb(part_path))
        except Exception:
            ok = False
    if not ok:
        _log("  complete resume buffer failed validation; downloading fresh")
        try:
            os.remove(part_path)
        except OSError as exc:
            # Never issue an EOF Range request or overwrite bytes that failed
            # authentication when the OS will not let us discard them.
            raise DownloadError(
                f"Couldn't discard a completed download that failed "
                f"validation: {exc}", terminal=True) from exc
        return False

    if status_cb is not None:
        try:
            status_cb("Finishing update…")
        except Exception:
            pass
    _finalize_download(part_path, dest_path)
    return True


# Backoff schedule (seconds) between retries.  Indexed by (attempt-1) and
# clamped to the last entry.  Deliberately stretches to ~30 s so a transient
# server-side 5xx burst — e.g. GitHub's release-download edge returning HTTP
# 504 for a minute or two right after a large asset is published — is ridden
# out instead of erroring.  A 504 comes back in ~0.2 s, so the wall-clock cost
# of an extra retry is almost entirely the backoff itself.
_RETRY_BACKOFFS = (2, 5, 10, 20, 30, 30)


def download_file(url: str,
                  dest_path: str,
                  *,
                  progress_cb: Optional[Callable[[int, int], None]] = None,
                  cancel_cb: Optional[Callable[[], bool]] = None,
                  validate_cb: Optional[Callable[[str], bool]] = None,
                  status_cb: Optional[Callable[[str], None]] = None,
                  headers: Optional[dict] = None,
                  max_attempts: int = 3,
                  timeout: float = 20.0,
                  read_stall_s: float = 10.0,
                  parallel_segments: int = 4,
                  expected_size: Optional[int] = None) -> None:
    """Download ``url`` → ``dest_path`` with progress, cancel, resume,
    validation and retry/backoff.

    Args:
      progress_cb(downloaded, total): called (throttled to ~10 Hz) with
        byte counts; ``total`` is 0 if the server sent no Content-Length.
      cancel_cb() -> bool: polled between chunks; return True to abort.
      validate_cb(path) -> bool: called on the finished ``.part`` before
        the atomic rename; return False to reject (treated as a corrupt
        download and retried).
      status_cb(msg): called with a human-readable status line during
        retry backoff (e.g. "Server busy — retrying in 12s…") so a
        watching UI stays informative while waiting out a transient 5xx.
      headers: extra request headers (merged over a default User-Agent).
      max_attempts: total tries before giving up (backoff 2/5/10 s).
      expected_size: authoritative byte size for this exact asset.  When a
        resumable ``.part`` already has this size, validate and atomically
        finalize it locally instead of requesting an unsatisfiable EOF range.

    Raises ``DownloadError`` on failure; returns None on success (the
    complete file is at ``dest_path``).
    """
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    if expected_size is not None:
        try:
            expected_size = int(expected_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected_size must be an integer byte count") from exc
        if expected_size <= 0:
            expected_size = None

    # Enforce the authoritative size on every path, including the parallel
    # downloader whose total normally comes from Content-Range.  This wrapper
    # also lets complete-part recovery reuse the ordinary validation contract.
    caller_validate_cb = validate_cb
    if expected_size is not None:
        def _validate_expected_size(path: str) -> bool:
            try:
                if os.path.getsize(path) != expected_size:
                    return False
            except OSError:
                return False
            return (caller_validate_cb is None
                    or bool(caller_validate_cb(path)))
        effective_validate_cb = _validate_expected_size
    else:
        effective_validate_cb = caller_validate_cb

    # Check before the parallel probe as well as before single-stream GETs: a
    # complete authenticated resume buffer requires no network request at all.
    if _recover_complete_part(
            dest_path, expected_size,
            progress_cb=progress_cb, cancel_cb=cancel_cb,
            validate_cb=effective_validate_cb, status_cb=status_cb):
        return

    # ── Fast path: parallel byte-range segments ──────────────────────────────
    # A CDN like GitHub's release edge throttles per-connection, so several
    # connections aggregate to markedly higher throughput than one stream.
    # Tried once up front; falls through to the single-stream retry loop below
    # on anything unexpected (no Range support, small file, a transient error).
    # Disable with FIREFLY_NO_PARALLEL_DOWNLOAD=1.
    if (parallel_segments and parallel_segments > 1
            and str(os.environ.get("FIREFLY_NO_PARALLEL_DOWNLOAD", "")).strip()
            not in ("1", "true", "yes", "on")):
        try:
            if _download_parallel(
                    url, dest_path, segments=int(parallel_segments),
                    progress_cb=progress_cb, cancel_cb=cancel_cb,
                    validate_cb=effective_validate_cb, headers=headers,
                    timeout=timeout,
                    status_cb=status_cb):
                return
        except DownloadError as exc:
            _m = str(exc).lower()
            if getattr(exc, "terminal", False) or "cancel" in _m or "http 4" in _m:
                raise                          # user cancel / permanent 4xx / terminal
            _log(f"  parallel download failed ({exc}); single-stream fallback")
        except Exception as exc:
            _log(f"  parallel download error "
                 f"({type(exc).__name__}: {exc}); single-stream fallback")

    # Retry loop.  Backoff (seconds): 2, 5, 10 — rides out a transient
    # firewall hiccup without boring the user.  The .part file is kept
    # between attempts so a retry resumes rather than re-downloading.
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            _download_once(url, dest_path,
                           progress_cb=progress_cb, cancel_cb=cancel_cb,
                           validate_cb=effective_validate_cb, headers=headers,
                           attempt=attempt, max_attempts=max_attempts,
                           timeout=timeout, read_stall_s=read_stall_s,
                           status_cb=status_cb, expected_size=expected_size)
            return
        except DownloadError as exc:
            # User-cancel + permanent server errors (4xx) + terminal failures
            # (e.g. the OS denied the final rename) propagate immediately —
            # don't burn retries re-downloading when a retry can't help.
            msg = str(exc).lower()
            if getattr(exc, "terminal", False) or "cancel" in msg or "http 4" in msg:
                raise
            last_exc = exc
            _log(f"  attempt {attempt}/{max_attempts} failed: {exc}")
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError) as exc:
            last_exc = exc
            _log(f"  attempt {attempt}/{max_attempts} failed "
                 f"({type(exc).__name__}: {exc})")
        if attempt < max_attempts:
            backoff = _RETRY_BACKOFFS[min(attempt - 1,
                                          len(_RETRY_BACKOFFS) - 1)]
            _log(f"  backing off {backoff}s before retry "
                 f"(resume buffer kept)")
            # Count down second-by-second so a watching UI sees the app is
            # alive (and can cancel) during a long backoff, and honour
            # cancellation promptly.
            for rem in range(backoff, 0, -1):
                if cancel_cb is not None:
                    try:
                        if cancel_cb():
                            raise DownloadError("Download cancelled by user.")
                    except DownloadError:
                        raise
                    except Exception:
                        pass
                if status_cb is not None:
                    try:
                        status_cb(f"Server busy — retrying in {rem}s… "
                                  f"(attempt {attempt + 1}/{max_attempts})")
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
                   read_stall_s: float,
                   status_cb: Optional[Callable[[str], None]] = None,
                   expected_size: Optional[int] = None) -> None:
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

    # A retry can begin with all bytes present if the preceding connection or
    # process failed after its final write but before validation/finalization.
    if _recover_complete_part(
            dest_path, expected_size,
            progress_cb=progress_cb, cancel_cb=cancel_cb,
            validate_cb=validate_cb, status_cb=status_cb):
        return
    try:
        resume_from = (int(os.path.getsize(part_path))
                       if os.path.exists(part_path) else 0)
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
        #    page that slipped through as a 200).  This can take a while — a
        #    SHA-256 pass over a big installer + (macOS) an hdiutil header read —
        #    so announce it: otherwise the bar sits at 100% looking frozen.
        if validate_cb is not None:
            if status_cb is not None:
                try: status_cb("Verifying download…")
                except Exception: pass
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
        # 3) Atomic finalize — only if every earlier check passed.  Retries a
        #    transient Windows file lock (Defender scanning the fresh .exe) and
        #    fails TERMINALLY if the OS keeps denying access (AV / policy).
        if status_cb is not None:
            try: status_cb("Finishing update…")
            except Exception: pass
        _finalize_download(part_path, dest_path)
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


def _download_parallel(url: str,
                       dest_path: str,
                       *,
                       segments: int,
                       progress_cb: Optional[Callable[[int, int], None]] = None,
                       cancel_cb: Optional[Callable[[], bool]] = None,
                       validate_cb: Optional[Callable[[str], bool]] = None,
                       headers: Optional[dict] = None,
                       timeout: float = 20.0,
                       status_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Download ``url`` in ``segments`` parallel byte-range requests, for higher
    throughput from a CDN that throttles per connection (GitHub releases).

    Returns True on success (complete file at ``dest_path``).  Returns False to
    tell the caller to fall back to the single-stream path — no Range support,
    file too small to bother, or any non-cancel error.  Raises ``DownloadError``
    only on user cancel.  Each segment streams to its own ``.part.segN`` file
    (bounded RAM), then they're concatenated, size-checked, validated and
    atomically renamed — mirroring the single-stream path's integrity gauntlet.
    """
    import concurrent.futures

    MIN_PARALLEL_BYTES = 8 * 1024 * 1024     # below this, one stream is fine
    CHUNK = 256 * 1024

    base_headers = {"User-Agent": "FIREFLY-app"}
    if headers:
        base_headers.update(headers)

    part_path = dest_path + ".part"

    def _seg_path(i):
        return f"{part_path}.seg{i}"

    def _cleanup():
        for i in range(64):                  # generous upper bound
            p = _seg_path(i)
            if not os.path.exists(p):
                if i >= int(segments):
                    break
                continue
            try:    os.remove(p)
            except Exception: pass

    # ── Probe: total size + Range support, via a 1-byte range GET ──
    try:
        ph = dict(base_headers); ph["Range"] = "bytes=0-0"
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=ph), timeout=timeout) as resp:
            status = int(getattr(resp, "status", 0) or 0)
            crange = resp.headers.get("Content-Range") or ""
        if status != 206 or "/" not in crange:
            return False                     # server ignored Range → fall back
        total = int(crange.rsplit("/", 1)[-1])
    except urllib.error.HTTPError as exc:
        code = getattr(exc, "code", 0)
        if 400 <= code < 500:
            # Permanent (404 etc.) — don't burn the single-stream retries on it.
            raise DownloadError(f"HTTP {code} when downloading {url}") from exc
        return False                         # 5xx → fall back (it retries)
    except Exception:
        return False
    if total < MIN_PARALLEL_BYTES:
        return False

    n = max(2, min(int(segments), 8))
    seg = total // n
    ranges = [(i, i * seg, (total - 1 if i == n - 1 else (i + 1) * seg - 1))
              for i in range(n)]
    _log(f"  parallel download: {n} segments, {total/1e6:.0f} MB total")

    lock = threading.Lock()
    state = {"last_t": 0.0, "cancel": False}
    seg_bytes = [0] * n                    # ACTUAL bytes on disk per segment
    _RANGE_IGNORED = "__range_ignored__"   # a 200 to a ranged request → fall back

    def _report():
        # Progress is the actual bytes on disk summed across segments, CAPPED at
        # the total — never a running chunk tally.  So a resumed / retried segment
        # can't double-count and push the bar past 100% (the bug users hit).
        if progress_cb is None:
            return
        with lock:
            now = time.monotonic()
            if now - state["last_t"] < 0.1:
                return
            state["last_t"] = now
            snap = min(sum(seg_bytes), total)
        try:    progress_cb(snap, total)
        except Exception: pass

    def _fetch(idx, start, end):
        # Retry + RESUME this one segment instead of letting a single dropped
        # connection fail the whole parallel download (the "downloads twice" users
        # saw on flaky / proxied / AV networks).  We resume from the bytes already
        # on disk, so a transient drop only re-fetches the missing tail.
        seg = _seg_path(idx)
        need = end - start + 1                    # bytes this segment must hold
        SEG_TRIES = 4
        with lock:
            seg_bytes[idx] = os.path.getsize(seg) if os.path.exists(seg) else 0
        for attempt in range(1, SEG_TRIES + 1):
            got = os.path.getsize(seg) if os.path.exists(seg) else 0
            if got >= need:
                with lock: seg_bytes[idx] = need
                _report()
                return                            # already complete
            h = dict(base_headers)
            h["Range"] = f"bytes={start + got}-{end}"   # resume from what we have
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers=h), timeout=timeout) as resp:
                    # A ranged request MUST come back 206.  A 200 means the server /
                    # proxy ignored Range and is sending the WHOLE file, not this
                    # segment — appending it would DUPLICATE bytes (bar past 100%)
                    # and corrupt the segment.  Bail to the single-stream path,
                    # which downloads a plain 200 correctly.
                    if getattr(resp, "status", 200) != 206:
                        raise DownloadError(_RANGE_IGNORED)
                    # Read ONLY this segment's own bytes.  A CDN / proxy that
                    # honours the range START but ignores the END answers a 206 by
                    # streaming to EOF (legal per RFC 7233).  Without this bound
                    # every non-last segment would swallow the whole tail of the
                    # file, so the assembled size overshoots `total`, the post-
                    # concat size check fails, and the parallel path falls back to
                    # a FULL single-stream re-download — the visible "downloads to
                    # 100%, then downloads all over again" bug.
                    remaining = need - got
                    with open(seg, "ab" if got else "wb") as out:
                        while remaining > 0:
                            if state["cancel"]:
                                raise DownloadError("Download cancelled by user.")
                            chunk = resp.read(min(CHUNK, remaining))
                            if not chunk:
                                break
                            out.write(chunk)
                            remaining -= len(chunk)
                            with lock:
                                seg_bytes[idx] += len(chunk)
                            _report()
                if os.path.getsize(seg) >= need:
                    with lock: seg_bytes[idx] = need
                    _report()
                    return                        # segment complete
                # short read (connection closed early) → loop to resume the tail
            except DownloadError:
                raise                             # cancel / range-ignored → propagate
            except Exception:
                if attempt >= SEG_TRIES:
                    raise                         # give up → parallel falls back once
                time.sleep(0.4 * attempt)         # brief backoff, then resume
        raise DownloadError(f"segment {idx} incomplete after {SEG_TRIES} tries")

    # Poll the cancel callback off the worker threads.
    stop_poll = threading.Event()

    def _poll():
        while not stop_poll.wait(0.3):
            if cancel_cb is not None:
                try:
                    if cancel_cb():
                        state["cancel"] = True
                        return
                except Exception:
                    pass
    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(_fetch, i, s, e) for i, s, e in ranges]
            for f in concurrent.futures.as_completed(futs):
                f.result()                   # raises on segment failure
    except DownloadError as exc:
        stop_poll.set(); _cleanup()
        if state["cancel"] or getattr(exc, "terminal", False):
            raise                            # user cancel / terminal → propagate
        return False
    except Exception:
        stop_poll.set(); _cleanup()
        return False
    finally:
        stop_poll.set()

    # ── Concatenate → .part, size-check, validate, atomic rename ──
    # Copy EXACTLY each segment's own byte span (not the whole file on disk), so a
    # segment left oversize by an end-ignoring server (or a prior buggy run) can't
    # push the assembled size past `total`.  Assembly is therefore always exactly
    # `total` bytes as long as each segment holds at least its span.
    try:
        with open(part_path, "wb") as out:
            for (i, s, e) in ranges:
                need_i = e - s + 1
                with open(_seg_path(i), "rb") as sp:
                    left = need_i
                    while left > 0:
                        buf = sp.read(min(4 * 1024 * 1024, left))
                        if not buf:
                            break
                        out.write(buf)
                        left -= len(buf)
                if left > 0:                     # segment short on disk → can't assemble
                    raise OSError(f"segment {i} short by {left} bytes")
    except Exception:
        _cleanup()
        try: os.remove(part_path)
        except Exception: pass
        return False
    _cleanup()

    try:
        actual = os.path.getsize(part_path)
    except Exception:
        actual = -1
    if actual != total:
        try: os.remove(part_path)
        except Exception: pass
        return False

    if validate_cb is not None:
        # Announce the (potentially slow) hash + format check so the bar doesn't
        # look frozen at 100% while it runs.
        if status_cb is not None:
            try: status_cb("Verifying download…")
            except Exception: pass
        try:    valid = bool(validate_cb(part_path))
        except Exception: valid = False
        if not valid:
            try: os.remove(part_path)
            except Exception: pass
            raise DownloadError(
                "Downloaded file failed validation (unexpected format — an "
                "intercepting proxy may have returned an error page).")

    if progress_cb is not None:
        try:    progress_cb(total, total)
        except Exception: pass
    # Same robust finalize as the single-stream path — ride out a transient
    # Windows file lock, fail terminally if the OS keeps denying access.
    if status_cb is not None:
        try: status_cb("Finishing update…")
        except Exception: pass
    _finalize_download(part_path, dest_path)
    _log(f"  ✓ parallel download complete: {total/1e6:.0f} MB")
    return True
