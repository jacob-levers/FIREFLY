"""
FIREFLY CUDA-torch sidecar installer.

The Windows .exe ships with a CPU-only PyTorch build because the CUDA
torch wheel (~2.5 GB) exceeds GitHub Releases' 2 GiB asset cap.  On
first launch on Windows we detect an NVIDIA GPU and offer to download
the matching CUDA torch wheel into %LOCALAPPDATA%\\FIREFLY\\torch-cuda
on demand.  On subsequent launches we prepend the extracted sidecar to
sys.path so `import torch` resolves to the CUDA build, shadowing the
bundled CPU build.

This module is pure stdlib — no PySide6 dependency — so it can be
imported safely from anywhere in the app (including before the Qt
event loop exists and before any torch import).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable, Optional


# ── Diagnostic log plumbing ───────────────────────────────────────────────────
# When a user reports "it gets stuck", we need a step-by-step breadcrumb
# trail of what the installer was doing.  Modules outside cuda_installer
# can register a callback via set_log_callback(); every call to _log()
# inside this module forwards there in addition to stdout.
_log_cb: Optional[Callable[[str], None]] = None
_log_t0: float = 0.0

# Last underlying reason a url_exists() probe failed (SSL error, timeout, …).
# Surfaced in install_cuda_torch_auto's error so the windowed .exe — which has
# no console and no longer shows a debug-log window — can report WHY the wheel
# check failed instead of a misleading "no wheel exists".
_last_probe_error: Optional[str] = None


def set_log_callback(cb: Optional[Callable[[str], None]]) -> None:
    """Register a callable that receives each diagnostic line.  Pass
    None to clear.  Lines are also always printed to stdout."""
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


# ── Platform helpers ──────────────────────────────────────────────────────────
def is_windows() -> bool:
    return sys.platform == "win32"


def _no_window_kwargs() -> dict:
    """subprocess kwargs that suppress the brief cmd.exe flash on Windows."""
    if not is_windows():
        return {}
    # CREATE_NO_WINDOW = 0x08000000 — defined in subprocess only on Windows
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return {"creationflags": flags}


# ── GPU detection ─────────────────────────────────────────────────────────────
def detect_nvidia_gpu() -> Optional[str]:
    """Return the first NVIDIA GPU name reported by nvidia-smi, or None.

    Uses a 5 s timeout.  Suppresses the cmd.exe window flash on Windows.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            **_no_window_kwargs(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    name = line[0].strip()
    return name or None


# ── Filesystem layout ─────────────────────────────────────────────────────────
def _current_py_tag() -> str:
    """CPython ABI tag for the running interpreter, e.g. 'cp313'.  A torch
    wheel's compiled extensions are tagged with this, and only load under a
    matching interpreter."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _sidecar_abi_ok(extracted: str) -> bool:
    """True if the extracted torch's compiled core extension matches THIS
    interpreter's ABI (e.g. _C.cp313-win_amd64.pyd under Python 3.13).

    A CUDA wheel installed under one Python version (say the old 3.12 build)
    must never be injected into a different one (the 3.13 build): the cp312
    `_C` binary fails to import under 3.13 and shadows the bundled torch,
    leaving the whole process unable to `import torch`.  This guard is what
    lets the version transition self-heal."""
    try:
        tdir = os.path.join(extracted, "torch")
        if not os.path.isdir(tdir):
            return False
        tag = _current_py_tag()
        found_ext = False
        for name in os.listdir(tdir):
            if name.startswith("_C.") and name.endswith((".pyd", ".so")):
                found_ext = True
                if f".{tag}-" in name or f".{tag}." in name:
                    return True
        # If a tagged _C extension exists but none matched our tag, it's for
        # a different interpreter — reject.  If no tagged _C was found at all
        # (unexpected layout), be permissive rather than block a real install.
        return not found_ext
    except Exception:
        return False


def sidecar_dir() -> str:
    """Per-interpreter sidecar root, e.g.
    %LOCALAPPDATA%\\FIREFLY\\torch-cuda\\cp313 on Windows,
    ~/.firefly/torch-cuda/cp313 elsewhere (dev/testing only).

    Version-namespaced by interpreter ABI tag so a CUDA wheel installed under
    one Python version is never picked up by a build on another — that
    cross-version collision is what broke `import torch` after the 3.13 bump.
    Parent dirs are created on demand."""
    if is_windows():
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        path = os.path.join(base, "FIREFLY", "torch-cuda", _current_py_tag())
    else:
        path = os.path.join(os.path.expanduser("~"), ".firefly",
                            "torch-cuda", _current_py_tag())
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def sidecar_extracted_dir() -> str:
    return os.path.join(sidecar_dir(), "extracted")


def is_installed() -> bool:
    """True only if a torch sidecar exists AND its ABI matches this
    interpreter.  The ABI check keeps the GUI honest: a mismatched leftover
    install reports as not-installed so the setup button reappears instead of
    silently poisoning `import torch`."""
    try:
        extracted = sidecar_extracted_dir()
        if not os.path.isfile(
                os.path.join(extracted, "torch", "__init__.py")):
            return False
        return _sidecar_abi_ok(extracted)
    except Exception:
        return False


# ── User-declined flag ────────────────────────────────────────────────────────
def settings_path() -> str:
    return os.path.join(sidecar_dir(), "state.json")


def _read_state() -> dict:
    try:
        with open(settings_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
        with open(settings_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except Exception:
        pass


def user_declined() -> bool:
    return bool(_read_state().get("declined", False))


def mark_declined() -> None:
    state = _read_state()
    state["declined"] = True
    _write_state(state)


def clear_declined() -> None:
    state = _read_state()
    state.pop("declined", None)
    _write_state(state)


# ── Torch version / URL building ──────────────────────────────────────────────
def bundled_torch_version() -> Optional[str]:
    """Return the base version of the currently-imported torch (e.g. '2.5.1'),
    stripping any '+cpu' / '+cu124' local-version suffix.  None if torch
    can't be imported."""
    try:
        import torch  # noqa: F401  — safe; we just read __version__
        ver = getattr(torch, "__version__", "") or ""
    except Exception:
        return None
    if not ver:
        return None
    # PEP 440 local-version separator is '+'
    base = ver.split("+", 1)[0].strip()
    return base or None


def cuda_wheel_url(torch_version: str, cuda_tag: str = "cu124",
                   python_tag: Optional[str] = None) -> str:
    """Build the CUDA wheel URL on download.pytorch.org.

    Example:
        torch_version='2.5.1', cuda_tag='cu124', python_tag='cp312'
        → https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl
    """
    if python_tag is None:
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # '+' must be URL-encoded as %2B in the local-version segment.
    filename = (
        f"torch-{torch_version}%2B{cuda_tag}"
        f"-{python_tag}-{python_tag}-win_amd64.whl"
    )
    return f"https://download.pytorch.org/whl/{cuda_tag}/{filename}"


# ── Download / extract ────────────────────────────────────────────────────────
def _download_via_bits(url: str,
                        dest_path: str,
                        progress_cb: Optional[Callable[[int, int], None]] = None,
                        cancel_cb: Optional[Callable[[], bool]] = None) -> None:
    """Windows-native download via PowerShell's BITS (Background
    Intelligent Transfer Service).  Used in preference to urllib on
    Windows because:

      * BITS uses the Windows HTTP stack (winhttp) → same code path as
        Windows Update → far more compatible with Defender real-time
        scanning, corporate proxies, and TLS deep-packet inspection
        appliances than Python's urllib.
      * Resumable across network blips.
      * The BITS service runs in the background — even if our process
        is suspended by Defender's scan, BITS keeps going.

    Drives a synchronous Start-BitsTransfer in PowerShell and polls the
    destination file size for progress (Get-BitsTransfer adds complexity
    for marginal gain over file-size polling).
    """
    import json as _json

    _log("Using Windows BITS (Background Intelligent Transfer Service)")
    # Inline PowerShell driver — start an async BITS job and emit
    # JSON progress to stdout every 500 ms until done.
    ps_script = r"""
$ErrorActionPreference = 'Stop'
Import-Module BitsTransfer
$src = $args[0]
$dst = $args[1]
# Async so we can stream progress AND honour cancellation by
# polling Python's stdin.  Synchronous mode would block until done
# with no way to monitor.
$job = $null
try {
    $job = Start-BitsTransfer -Source $src -Destination $dst `
                              -Asynchronous `
                              -DisplayName 'FIREFLY-CUDA-installer' `
                              -Priority Foreground
    while ($true) {
        Start-Sleep -Milliseconds 500
        $j = Get-BitsTransfer -JobId $job.JobId -ErrorAction SilentlyContinue
        if ($null -eq $j) { break }
        $payload = @{
            state = [string]$j.JobState
            transferred = [int64]$j.BytesTransferred
            total = [int64]$j.BytesTotal
        }
        Write-Output ($payload | ConvertTo-Json -Compress)
        switch ($j.JobState) {
            'Transferred' {
                Complete-BitsTransfer -BitsJob $j
                Write-Output '{"state":"Done","transferred":0,"total":0}'
                return
            }
            'Error' {
                $err = $j.ErrorDescription
                Remove-BitsTransfer -BitsJob $j -ErrorAction SilentlyContinue
                throw "BITS Error: $err"
            }
            'Cancelled' {
                Remove-BitsTransfer -BitsJob $j -ErrorAction SilentlyContinue
                throw 'BITS Cancelled'
            }
        }
    }
} catch {
    if ($job) {
        Remove-BitsTransfer -BitsJob $job -ErrorAction SilentlyContinue
    }
    throw
}
"""

    # Use -EncodedCommand so quoting around the URL is bulletproof.
    import base64 as _b64
    encoded = _b64.b64encode(ps_script.encode("utf-16le")).decode("ascii")

    # Open PowerShell, pipe stdout to us so we can stream progress.
    creationflags = 0
    if is_windows():
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    # NOTE: do NOT pass "--" between -EncodedCommand and the positional
    # args.  powershell.exe is not POSIX getopt: it interprets "--" as
    # an ambiguous prefix of "-Command" and responds by dumping the
    # help text for -Command to stdout, then exiting WITHOUT running
    # our script.  We previously hit that here — the help text included
    # quoted strings that happened to be valid JSON literals, so
    # json.loads() returned a `str`, and ev.get(...) below raised
    # `AttributeError("'str' object has no attribute 'get'")`, which
    # bubbled up to download_wheel() as "BITS path unavailable" and
    # forced a urllib fallback on every Windows install.  Anything
    # after -EncodedCommand <base64> is already forwarded to $args in
    # the embedded script — no separator required.
    cmd = ["powershell", "-NoProfile", "-NonInteractive",
           "-EncodedCommand", encoded,
           url, dest_path]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )

    last_transferred = 0
    no_progress_since = time.monotonic()
    stall_limit_s = 30.0   # BITS handles transient failures internally —
                            # only abort if it's COMPLETELY stuck.
    # Separate, shorter timeout for the "BITS hasn't moved a byte yet"
    # phase.  BITS likes to sit in Connecting/Transferring with
    # transferred=0, total=0 indefinitely when something upstream is
    # wrong (corporate proxy, blocked port, Defender scanning).  Users
    # see "Downloading… 0 MB" with no movement and reasonably assume
    # the app's frozen.  After this many seconds with zero bytes
    # actually delivered we BAIL on BITS — but with a non-RuntimeError
    # so `download_wheel` falls back to urllib instead of failing the
    # whole install (RuntimeError is reserved for user-visible BITS
    # errors we don't want masked by the fallback).
    initial_connect_timeout_s = 12.0
    started_at = time.monotonic()

    try:
        while True:
            if cancel_cb is not None:
                try:
                    if cancel_cb():
                        _log("  → Cancel requested, killing PowerShell + BITS job")
                        proc.terminate()
                        try: proc.wait(timeout=5)
                        except Exception: pass
                        try: os.remove(dest_path)
                        except Exception: pass
                        raise RuntimeError("Download cancelled by user.")
                except RuntimeError:
                    raise
                except Exception:
                    pass

            line = proc.stdout.readline()
            if not line:
                # stdout closed → PowerShell exited
                break
            line = line.strip()
            if not line:
                continue
            try:
                ev = _json.loads(line)
            except Exception:
                # Non-JSON output (warnings etc.) — just log it
                _log(f"  ps: {line}")
                continue
            # Defensive: json.loads happily returns a str/int/list if
            # PowerShell prints a bare JSON literal (e.g. its own help
            # text contains quoted strings).  We expect an object —
            # anything else is treated like the non-JSON branch above.
            if not isinstance(ev, dict):
                _log(f"  ps: {line}")
                continue
            state = ev.get("state", "")
            trans = int(ev.get("transferred", 0))
            total = int(ev.get("total", 0))
            if state == "Done":
                _log("  ✓ BITS reports Transfer complete")
                break
            # Stall detection — BITS has internal retries but if it's
            # genuinely dead, fail fast rather than wait forever.
            if trans > last_transferred:
                last_transferred = trans
                no_progress_since = time.monotonic()
            elif (last_transferred == 0
                    and (time.monotonic() - started_at)
                        > initial_connect_timeout_s):
                # BITS is wedged in Connecting/Transferring with zero
                # bytes received.  Most common cause on user reports:
                # Defender intercepting the TLS handshake or a
                # corporate proxy holding the connection.  Bail and
                # let urllib have a go — its read() blocks differently
                # and often gets through where BITS doesn't.
                elapsed = time.monotonic() - started_at
                _log(f"  → BITS hasn't received any bytes after "
                     f"{elapsed:.0f}s (state={state}); abandoning BITS "
                     f"and falling back to urllib")
                try:    proc.terminate()
                except Exception: pass
                # Plain Exception (not RuntimeError) so download_wheel's
                # `except RuntimeError: raise` does NOT catch it — the
                # broader `except Exception` falls through to urllib.
                raise Exception(
                    f"BITS failed to start transferring within "
                    f"{initial_connect_timeout_s:.0f}s")
            elif time.monotonic() - no_progress_since > stall_limit_s:
                _log(f"  → BITS stalled: no progress for {stall_limit_s:.0f}s "
                     f"({trans/1e6:.1f} MB / {total/1e6:.1f} MB)")
                proc.terminate()
                raise RuntimeError(
                    f"BITS download stalled at {trans/1e6:.1f} MB "
                    f"of {total/1e6:.1f} MB.  Check your network "
                    f"connection or try again later.")
            if progress_cb is not None:
                try:
                    progress_cb(trans, total)
                except Exception:
                    pass
            # Per-2s heartbeat into the debug log so the user sees activity
            # without being flooded.
            if int(time.monotonic() * 2) % 4 == 0:
                _log(f"  BITS: {state} {trans/1e6:.1f} / {total/1e6:.1f} MB")

        rc = proc.wait(timeout=10)
        stderr = proc.stderr.read() or ""
        if rc != 0:
            raise RuntimeError(
                f"PowerShell/BITS exited with code {rc}: {stderr.strip()}")

        if not os.path.exists(dest_path):
            raise RuntimeError(
                "BITS completed but the destination file is missing.")
        size_mb = os.path.getsize(dest_path) / 1e6
        _log(f"  ✓ download complete via BITS: {size_mb:.1f} MB")
        if progress_cb is not None:
            try:
                progress_cb(int(size_mb * 1e6), int(size_mb * 1e6))
            except Exception:
                pass
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            pass


def download_wheel(url: str,
                   dest_path: str,
                   progress_cb: Optional[Callable[[int, int], None]] = None,
                   cancel_cb: Optional[Callable[[], bool]] = None,
                   max_attempts: int = 3) -> None:
    """Download `url` → `dest_path` with progress + cancel support.

    Bulletproofing (Murphy's Law mode):
      * Atomic write — bytes land in `<dest>.part`; rename to `dest`
        only after the full download verifies.  No partial file ever
        looks like a complete one.
      * Resumable — if `<dest>.part` is left over from a prior crash
        we resume with a `Range: bytes=N-` header instead of starting
        over.  Saves 2.5 GB of bandwidth on every retry.
      * Retry-with-backoff — transient network failures (URLError,
        TimeoutError, stall watchdog tripping, short reads) re-enter
        the attempt loop up to `max_attempts` times with exponential
        backoff.  Permanent failures (HTTP 4xx, user cancel) abort
        immediately.
      * BITS demoted to opt-in — historically caused more hangs than
        it prevented (Defender / corp-proxy compatibility was the
        original justification, but urllib turns out to be just as
        compatible and we control the timeout behaviour).  Set the
        env var `FIREFLY_USE_BITS=1` to opt back in.
      * Zip-validity check — once the download finishes, the file is
        opened with zipfile.is_zipfile; if it's not a zip we retry
        rather than handing extract_wheel a corrupt archive.
    """
    # BITS path: opt-in only via env var.  The default fall-through is
    # urllib with retry/resume below — every problem we've seen on
    # Windows traces to BITS's opaque state machine, not urllib.
    if is_windows() and os.environ.get("FIREFLY_USE_BITS") == "1":
        try:
            _download_via_bits(url, dest_path,
                                progress_cb=progress_cb,
                                cancel_cb=cancel_cb)
            return
        except RuntimeError:
            # Re-raise user-cancel and BITS-specific errors as-is.
            raise
        except Exception as exc:
            _log(f"  BITS path unavailable ({exc!r}); falling back "
                 f"to urllib")
            # fall through to urllib path
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    # Retry loop.  Backoff schedule (seconds): 2, 5, 10 — enough to ride
    # out a transient firewall hiccup but not enough to bore the user.
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            _download_wheel_once(url, dest_path,
                                  progress_cb=progress_cb,
                                  cancel_cb=cancel_cb,
                                  attempt=attempt,
                                  max_attempts=max_attempts)
            return
        except RuntimeError as exc:
            # User cancel + permanent server errors are RuntimeError
            # and propagate immediately — don't burn retries on a 404.
            msg = str(exc).lower()
            if ("cancel" in msg or "http 4" in msg or "no cuda wheel" in msg):
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
                 f"(part file kept for resume)")
            for _ in range(backoff):
                if cancel_cb is not None:
                    try:
                        if cancel_cb():
                            raise RuntimeError("Download cancelled by user.")
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                time.sleep(1)
    # All attempts exhausted — clean up the .part and report failure.
    part_path = dest_path + ".part"
    try:
        if os.path.exists(part_path): os.remove(part_path)
    except Exception:
        pass
    raise RuntimeError(
        f"CUDA wheel download failed after {max_attempts} attempts.  "
        f"Last error: {last_exc}\n\n"
        f"Try again later, switch networks, or install FIREFLY from "
        f"source (README → 'Enabling CUDA') to use pip's downloader "
        f"instead."
    )


def _download_wheel_once(url: str,
                          dest_path: str,
                          *,
                          progress_cb: Optional[Callable[[int, int], None]] = None,
                          cancel_cb: Optional[Callable[[], bool]] = None,
                          attempt: int = 1,
                          max_attempts: int = 1) -> None:
    """One attempt at downloading the wheel.  Writes to `<dest>.part`,
    resumes from any prior `<dest>.part`, atomic-renames to `dest` on
    success.  Raises on any failure — `download_wheel` decides whether
    to retry.
    """
    part_path = dest_path + ".part"
    # Resume support — if a previous attempt left bytes behind, send a
    # Range header and append to the existing file.  download.pytorch.org
    # supports Range; if the server doesn't (we'd see HTTP 200 instead
    # of 206), we transparently fall back to a fresh download.
    resume_from = 0
    try:
        if os.path.exists(part_path):
            resume_from = int(os.path.getsize(part_path))
    except Exception:
        resume_from = 0
    # Bigger chunks → fewer read() syscalls AND fewer progress signals
    # queued to the GUI thread.  At ~50 MB/s on a 64 KB chunk that's
    # ~800 signal emissions per second, which overwhelms Qt's event
    # queue and starves paint events — Windows then marks the app
    # "Not Responding" even though the download is fine.  256 KB cuts
    # that to ~200/s and we additionally throttle progress_cb to
    # ~10 Hz below.
    chunk_size = 256 * 1024
    progress_throttle_s = 0.1   # 10 Hz cap on progress callbacks
    last_progress_t = 0.0
    # Remove any pre-existing FINAL file (we always re-derive it from
    # .part).  Leave the .part in place — that's our resume buffer.
    try:
        if os.path.exists(dest_path):
            os.remove(dest_path)
    except Exception:
        pass

    # Surface the URL in the FIREFLY console log — when the dialog
    # appears stuck, the user (and we) can read the log to see if the
    # URL itself is 404 (wrong torch version → no matching cu wheel)
    # vs a real network problem.
    _log(f"GET {url}  (attempt {attempt}/{max_attempts}"
         + (f", resume from {resume_from/1e6:.1f} MB" if resume_from else "")
         + ")")
    _log(f"  dest: {dest_path}  (writing to {os.path.basename(part_path)})")

    # Watchdog for stalled reads.  resp.read(N) can block forever on
    # Windows when the TLS connection stalls mid-stream (same bug class
    # that hung HEAD on cu118) or when Windows Defender / a corporate
    # firewall is intercepting the .whl write.  We sample `downloaded`
    # every 1 s in a daemon thread; if no bytes arrive for
    # `read_stall_s` seconds we tear the response down.  The worker's
    # read call then returns cleanly (or raises) and we fail with a
    # clear error instead of hanging the app forever.
    #
    # We ALSO emit an "activity heartbeat" log line every 2 s from the
    # same daemon thread so the debug-log window keeps updating while
    # the worker thread is blocked in resp.read() — without this, the
    # log appears frozen at "Starting read loop" and the user can't
    # tell whether anything is happening at all.
    import threading
    read_stall_s = 10.0
    progress_state = {"downloaded": 0, "last_change_t": time.monotonic(),
                       "should_abort": False, "done": False}
    resp_holder: dict = {"resp": None}

    def _stall_watchdog():
        wd_start = time.monotonic()
        last_heartbeat_at = wd_start
        last_reported_bytes = 0
        while not progress_state["should_abort"]:
            time.sleep(1.0)
            now = time.monotonic()
            elapsed = now - progress_state["last_change_t"]
            if progress_state.get("done"):
                return
            # Heartbeat every 2 s — proves the watchdog thread (and
            # therefore the Python interpreter / main loop) is alive,
            # and shows whether bytes are trickling in slowly.
            dl = progress_state["downloaded"]
            if now - last_heartbeat_at >= 2.0:
                last_heartbeat_at = now
                if dl == last_reported_bytes:
                    _log(f"  … still waiting for first chunk "
                         f"({elapsed:.0f}s since last activity)")
                else:
                    _log(f"  … downloading slowly: {dl/1e6:.1f} MB so far "
                         f"({(dl/1e6)/(now-wd_start):.2f} MB/s avg)")
                last_reported_bytes = dl
            if elapsed > read_stall_s:
                _log(f"  → STALL WATCHDOG: no data for {elapsed:.0f}s, "
                     f"aborting (downloaded {dl/1e6:.1f} MB)")
                progress_state["should_abort"] = True
                try:
                    r = resp_holder.get("resp")
                    if r is not None:
                        r.close()
                except Exception:
                    pass
                return

    try:
        headers = {"User-Agent": "FIREFLY-CUDA-installer/1.0"}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
        req = urllib.request.Request(url, headers=headers)
        # 20 s timeout (was 30) so a dead URL fails fast instead of
        # leaving the user staring at a frozen-looking dialog.
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp_holder["resp"] = resp
            status = int(getattr(resp, "status", 0) or 0)
            _log(f"  HTTP {status} in {time.monotonic()-t0:.2f}s")
            # If we asked for a range but the server returned 200, it's
            # ignoring our Range header — discard the .part and start
            # fresh.  Better to re-download than corrupt the file.
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
            # On a 206 Partial Content, Content-Length is the REMAINDER,
            # not the total.  Total file size = resume_from + remainder.
            total = (resume_from + cl) if (status == 206 and cl > 0) else cl
            _log(f"  Content-Length: {cl/1e6:.1f} MB"
                 + (f"   (total wheel: {total/1e6:.1f} MB)"
                    if status == 206 else ""))
            _log(f"  Starting read loop (chunk={chunk_size//1024} KB, "
                 f"stall watchdog={read_stall_s:.0f}s)")
            wdog = threading.Thread(target=_stall_watchdog, daemon=True,
                                     name="cuda-download-stall-watchdog")
            wdog.start()
            downloaded = resume_from
            # Seed the watchdog so the "no progress yet" check doesn't
            # immediately fire on resume.
            progress_state["downloaded"] = downloaded
            chunk_count = 0
            last_diag_t = time.monotonic()
            # Append-mode for resume, write-mode for fresh download.
            file_mode = "ab" if resume_from > 0 else "wb"
            with open(part_path, file_mode) as out:
                while True:
                    if cancel_cb is not None:
                        try:
                            if cancel_cb():
                                # Cancelled — leave .part in place so
                                # the user can resume by re-running the
                                # installer; only delete on full
                                # uninstall.
                                progress_state["should_abort"] = True
                                raise RuntimeError(
                                    "Download cancelled by user.")
                        except RuntimeError:
                            raise
                        except Exception:
                            pass
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    if progress_state["should_abort"]:
                        # Don't delete .part — let the retry loop resume.
                        raise RuntimeError(
                            "Download stalled — no data received from "
                            "download.pytorch.org for 10 seconds.")
                    out.write(chunk)
                    downloaded += len(chunk)
                    chunk_count += 1
                    # Diagnostic log: first 3 chunks individually (so we
                    # can see bytes ARE arriving), then once every 1 s.
                    now = time.monotonic()
                    if chunk_count <= 3 or (now - last_diag_t) >= 1.0:
                        last_diag_t = now
                        _log(f"  + chunk {chunk_count}: "
                             f"{downloaded/1e6:.1f} MB / {total/1e6:.0f} MB")
                    progress_state["downloaded"] = downloaded
                    progress_state["last_change_t"] = now
                    # Throttle progress emissions to ~10 Hz.  Without
                    # this, a fast connection floods the main Qt event
                    # queue with thousands of queued slot calls per
                    # second, paint events get starved, and Windows
                    # marks the app "Not Responding".
                    if progress_cb is not None:
                        if (now - last_progress_t) >= progress_throttle_s:
                            last_progress_t = now
                            try:
                                progress_cb(downloaded, total)
                            except Exception:
                                pass
                # Final 100 % tick so the bar visibly hits the end.
                progress_state["done"] = True
                _log(f"  ✓ download complete: {downloaded/1e6:.1f} MB "
                     f"in {time.monotonic()-t0:.1f}s "
                     f"({(downloaded/1e6)/(time.monotonic()-t0):.1f} MB/s)")
                if progress_cb is not None:
                    try:
                        progress_cb(downloaded, total)
                    except Exception:
                        pass

        # ── Post-download integrity gauntlet ─────────────────────────
        # 1) File size matches Content-Length (if the server sent one).
        try:
            actual = os.path.getsize(part_path)
        except Exception:
            actual = 0
        if total > 0 and actual != total:
            # Short read — keep the .part so the next attempt can
            # resume, but raise so the retry loop fires.
            raise RuntimeError(
                f"Short read: got {actual/1e6:.1f} MB but Content-Length "
                f"indicated {total/1e6:.1f} MB.")
        # 2) Looks like a valid zip (wheels are zips).  Catches the
        # case where a captive portal / proxy returned an HTML page
        # but the urllib stack didn't notice.
        try:
            ok_zip = zipfile.is_zipfile(part_path)
        except Exception:
            ok_zip = False
        if not ok_zip:
            # NOT a valid zip — almost certainly an HTML error page
            # or a transparent proxy page got captured.  Discard.
            try:    os.remove(part_path)
            except Exception: pass
            raise RuntimeError(
                "Downloaded file is not a valid .whl/.zip — likely an "
                "intercepting proxy returned an HTML error page.  "
                "Try again on a different network.")
        # 3) Atomic rename to the final path — only happens if every
        # earlier check passed.  os.replace is atomic on POSIX and
        # near-atomic on Windows (NTFS); either the final file exists
        # complete or it doesn't.
        try:
            os.replace(part_path, dest_path)
        except OSError as exc:
            # Most common on Windows: Defender has the .part open for
            # scanning.  Wait a beat and retry once before giving up.
            _log(f"  rename failed ({exc}); retrying after 1s "
                 f"(Defender scan?)")
            time.sleep(1.0)
            os.replace(part_path, dest_path)
    except urllib.error.HTTPError as exc:
        progress_state["should_abort"] = True
        # 416 = "Requested Range Not Satisfiable" — our .part is now
        # bigger than the server's file (rare, but happens when the
        # server's wheel was updated between attempts).  Discard.
        if getattr(exc, "code", 0) == 416:
            _log("  HTTP 416 — discarding stale .part and retrying fresh")
            try: os.remove(part_path)
            except Exception: pass
            raise urllib.error.URLError(f"416: {exc.reason}") from exc
        # 4xx other than 416 → permanent.  Tag with HTTP so the retry
        # loop sees it and aborts immediately instead of burning 3
        # attempts on a 404.
        if 400 <= getattr(exc, "code", 0) < 500:
            raise RuntimeError(
                f"HTTP {exc.code} when downloading {url}\n"
                f"The exact wheel build for this Python/torch "
                f"combination may not be on download.pytorch.org."
            ) from exc
        # 5xx → transient, let the retry loop have it.
        raise urllib.error.URLError(
            f"{exc.code}: {exc.reason}") from exc
    except urllib.error.URLError:
        # Let the outer retry loop catch this — keep .part for resume.
        raise
    except RuntimeError:
        # Includes user-cancel + our own short-read / stalled errors.
        raise
    except Exception as exc:
        # Unknown exception — convert so the retry loop sees it; keep
        # .part on disk for the next attempt to resume from.
        raise RuntimeError(
            f"Unexpected error while downloading the CUDA PyTorch wheel: "
            f"{exc}"
        ) from exc


def extract_wheel(wheel_path: str,
                  dest_dir: str,
                  progress_cb: Optional[Callable[[int, int], None]] = None
                  ) -> None:
    """Extract the .whl (a zip) into dest_dir.  progress_cb(done, total)
    is invoked at ~10 Hz.

    If extraction fails partway, the partial `dest_dir` is wiped before
    the exception propagates — leaving a half-extracted sidecar around
    confuses `is_installed()` (torch/__init__.py might be present even
    though the rest of the wheel is missing) and creates ImportErrors
    at runtime that look unrelated to the installer.
    """
    # If dest_dir already has contents (caller didn't clean up), don't
    # silently merge — extraction's atomicity contract is per-call.
    os.makedirs(dest_dir, exist_ok=True)
    last_t = 0.0
    progress_throttle_s = 0.1
    bad_zip_cleanup_needed = False
    try:
        # Quick integrity check before iterating — zipfile.is_zipfile
        # is constant-time (reads the central directory) and catches
        # the "captive portal returned HTML" case if download_wheel's
        # check somehow missed it.
        if not zipfile.is_zipfile(wheel_path):
            bad_zip_cleanup_needed = True
            raise zipfile.BadZipFile(
                f"{wheel_path} is not a valid zip archive.")
        with zipfile.ZipFile(wheel_path) as zf:
            names = zf.namelist()
            total = len(names)
            for i, name in enumerate(names, start=1):
                try:
                    zf.extract(name, dest_dir)
                except OSError as exc:
                    # On Windows: file-already-in-use, path-too-long,
                    # Defender-blocking-write.  Add the failing path
                    # to the error so the user can see what's blocked.
                    raise RuntimeError(
                        f"Could not write {name} during extraction: "
                        f"{exc}.\n\nIf this is a Defender-scanning "
                        f"issue, exclude {dest_dir} from real-time "
                        f"scanning and retry."
                    ) from exc
                if progress_cb is not None:
                    now = time.monotonic()
                    if (now - last_t) >= progress_throttle_s or i == total:
                        last_t = now
                        try:
                            progress_cb(i, total)
                        except Exception:
                            pass
    except zipfile.BadZipFile as exc:
        # Wheel is corrupt — delete it so the retry path on the next
        # invocation downloads fresh instead of resuming a broken
        # .part.  Also clean up the partial extraction.
        try:    os.remove(wheel_path)
        except Exception: pass
        if bad_zip_cleanup_needed or os.path.isdir(dest_dir):
            try:    shutil.rmtree(dest_dir, ignore_errors=True)
            except Exception: pass
        raise RuntimeError(
            "The downloaded CUDA PyTorch wheel is corrupt.  The bad "
            "file has been removed; please retry the installation."
        ) from exc
    except RuntimeError:
        # Our own pre-formatted error; clean up partial extraction
        # but leave the wheel for forensics.
        try:    shutil.rmtree(dest_dir, ignore_errors=True)
        except Exception: pass
        raise
    except Exception as exc:
        # Unknown — clean up partial extraction.
        try:    shutil.rmtree(dest_dir, ignore_errors=True)
        except Exception: pass
        raise RuntimeError(
            f"Could not extract the CUDA PyTorch wheel: {exc}"
        ) from exc


# ── End-to-end installer ──────────────────────────────────────────────────────
def install_cuda_torch(cuda_tag: str = "cu124",
                       download_progress_cb=None,
                       extract_progress_cb=None,
                       cancel_cb=None) -> None:
    """Download + extract the CUDA torch wheel matching the currently-
    bundled torch version into the sidecar directory.

    Thin wrapper that delegates to `install_cuda_torch_from_url` so the
    hardening (atomic rename, partial-extract cleanup, disk-space
    pre-flight, etc.) applies here too.

    Raises RuntimeError with user-facing wording on any failure.
    """
    ver = bundled_torch_version()
    if not ver:
        raise RuntimeError(
            "Could not determine the bundled PyTorch version.  Cannot "
            "install CUDA acceleration without a matching version.")

    url = cuda_wheel_url(ver, cuda_tag=cuda_tag)
    install_cuda_torch_from_url(
        url,
        torch_version=ver,
        cuda_tag=cuda_tag,
        download_progress_cb=download_progress_cb,
        extract_progress_cb=extract_progress_cb,
        cancel_cb=cancel_cb,
    )


def url_exists(url: str, timeout: float = 8.0) -> bool:
    """HEAD request to check whether a wheel URL is reachable.

    Wrapped in a hard wall-clock watchdog: urllib's `timeout` is not
    reliably honored on Windows when the TLS handshake or DNS stage
    stalls (observed: cu118 HEAD hung indefinitely on Windows 11).
    The watchdog runs the actual request on a daemon thread and gives
    up after `timeout + 2 s`, so a stuck HEAD can never wedge the
    worker thread (which was making the whole app look frozen).

    Returns True on 2xx, False on anything else.  Never raises.
    """
    _log(f"HEAD {url}")
    _log(f"  (timeout={timeout}s, watchdog={timeout + 2}s)")

    global _last_probe_error
    import threading
    result_holder = {"ok": False, "done": False, "err": None}
    t0 = time.monotonic()

    def _do_head():
        try:
            req = urllib.request.Request(
                url, method="HEAD",
                headers={"User-Agent": "FIREFLY-CUDA-installer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                dt = time.monotonic() - t0
                code = int(getattr(resp, "status", 0) or 0)
                _log(f"  → HTTP {code} in {dt:.2f}s")
                result_holder["ok"] = 200 <= code < 300
                if not result_holder["ok"]:
                    result_holder["err"] = f"HTTP {code}"
        except urllib.error.HTTPError as exc:
            _log(f"  → HTTPError {exc.code}: {exc.reason} "
                 f"in {time.monotonic()-t0:.2f}s")
            result_holder["ok"] = False
            result_holder["err"] = f"HTTP {exc.code} {exc.reason}"
        except urllib.error.URLError as exc:
            _log(f"  → URLError: {exc.reason} "
                 f"in {time.monotonic()-t0:.2f}s")
            result_holder["ok"] = False
            result_holder["err"] = f"connection failed: {exc.reason}"
        except Exception as exc:
            _log(f"  → {type(exc).__name__}: {exc} "
                 f"in {time.monotonic()-t0:.2f}s")
            result_holder["ok"] = False
            result_holder["err"] = f"{type(exc).__name__}: {exc}"
        finally:
            result_holder["done"] = True

    t = threading.Thread(target=_do_head, daemon=True,
                          name="cuda-head-watchdog")
    t.start()
    t.join(timeout=timeout + 2)
    if not result_holder["done"]:
        _log(f"  → WATCHDOG: HEAD hung past {timeout + 2}s "
             f"(urllib timeout not honored — likely Windows TLS "
             f"handshake stall).  Treating as unreachable, moving on.")
        # Daemon thread will keep running but won't block process exit
        # or the worker thread.  Critical: we DON'T close the socket
        # here — that would race with the daemon thread.  It'll time
        # out eventually and exit on its own.
        _last_probe_error = f"probe timed out after {timeout + 2:.0f}s"
        return False
    if result_holder["err"]:
        _last_probe_error = result_holder["err"]
    return result_holder["ok"]


def install_cuda_torch_auto(torch_version: str,
                             cuda_tags: tuple = ("cu124", "cu121", "cu118"),
                             download_progress_cb=None,
                             extract_progress_cb=None,
                             cancel_cb=None,
                             status_cb: Optional[Callable[[str], None]] = None
                             ) -> str:
    """Try each CUDA tag in `cuda_tags` until one is reachable, then
    download.  Returns the cuda_tag that worked.

    Strategy: cheap HEAD requests to pick the first tag whose wheel
    actually exists (each HEAD is <1 s on a normal connection), THEN
    one full GET.  Avoids the 60-second triple-timeout stall the user
    hit when the bundled torch version doesn't have any CUDA wheel.

    `status_cb(msg)` is called between attempts so the GUI can update
    its label ("Checking cu121…", "Found cu121, downloading…").
    """
    _log(f"install_cuda_torch_auto starting — torch_version={torch_version}, "
         f"cuda_tags={cuda_tags}")
    chosen_tag: Optional[str] = None
    tried_urls = []
    for tag in cuda_tags:
        url = cuda_wheel_url(torch_version, cuda_tag=tag)
        tried_urls.append(url)
        _log(f"--- Checking {tag} ---")
        if status_cb is not None:
            try: status_cb(f"Checking torch {torch_version} + {tag}…")
            except Exception: pass
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Installation cancelled by user.")
        if url_exists(url):
            _log(f"  ✓ {tag} is available, will download")
            chosen_tag = tag
            break
        _log(f"  ✗ {tag} not available, trying next")

    if chosen_tag is None:
        _log("✗ No CUDA tag returned a working wheel URL")
        # All three HEAD-checks said "not found" — make the failure
        # actionable instead of mysterious.  Most likely cause: the
        # bundled torch version isn't a real release on PyTorch's
        # index (e.g. a pre-release or test version).
        url_lines = "\n  ".join(tried_urls)
        reason = (f"\n\nLast probe error: {_last_probe_error}"
                  if _last_probe_error else "")
        raise RuntimeError(
            f"Couldn't reach a CUDA wheel for torch {torch_version} at "
            f"download.pytorch.org.{reason}\n\n"
            f"Tried:\n  {url_lines}\n\n"
            f"If the last error mentions a certificate/SSL or connection "
            f"problem, it's a network/proxy issue rather than a missing "
            f"wheel.  Otherwise the bundled torch version may be one "
            f"PyTorch hasn't shipped CUDA builds for — install FIREFLY "
            f"from source and follow the 'Enabling CUDA' section of the "
            f"README to let pip resolve a matching wheel."
        )

    url = cuda_wheel_url(torch_version, cuda_tag=chosen_tag)
    if status_cb is not None:
        try: status_cb(f"Found cu{chosen_tag[2:]}, downloading…")
        except Exception: pass
    install_cuda_torch_from_url(
        url, torch_version=torch_version, cuda_tag=chosen_tag,
        download_progress_cb=download_progress_cb,
        extract_progress_cb=extract_progress_cb,
        cancel_cb=cancel_cb)
    return chosen_tag


def _check_free_space_gb(target_dir: str, needed_gb: float) -> None:
    """Raise RuntimeError if the volume containing `target_dir` has
    less than `needed_gb` GB free.  Refusing to start beats running out
    of disk halfway through extraction and leaving a half-installed
    sidecar that the user has no clear way to recover.
    """
    try:
        free_bytes = shutil.disk_usage(target_dir).free
    except Exception:
        # Can't measure — don't enforce.  Better to try and fail than
        # to refuse to start on a working drive.
        return
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < needed_gb:
        raise RuntimeError(
            f"Not enough disk space.  CUDA installation needs about "
            f"{needed_gb:.1f} GB free on the drive containing "
            f"{target_dir}, but only {free_gb:.1f} GB is available.\n\n"
            f"Free up space (the wheel and its extraction together are "
            f"~5 GB) and try again."
        )


def install_cuda_torch_from_url(url: str,
                                 *,
                                 torch_version: str,
                                 cuda_tag: str = "cu124",
                                 download_progress_cb=None,
                                 extract_progress_cb=None,
                                 cancel_cb=None) -> None:
    """Same as install_cuda_torch() but with the wheel URL already
    resolved by the caller — avoids a second `import torch` (which is
    slow on Windows onefile bundles).  `torch_version` is only used
    to name the temporary .whl file.

    Murphy's-Law-grade lifecycle:
      1. Pre-flight: disk-space check (raises if < 6 GB free).
      2. Wipe stale `extracted.partial/` from any prior killed install.
      3. Wipe stale `extracted/` so we never half-overlay a new wheel
         on an old one (mixed torch versions = import errors at
         runtime that look unrelated to the install).
      4. Download to `<wheel>.part` with retry/resume/atomic-rename.
      5. Extract to `extracted.partial/`; atomic-rename to `extracted/`
         on success.
      6. Delete the .whl (free ~2.5 GB).

    Any step that fails leaves the system in a recoverable state: the
    .part can be resumed, the partial extraction dir is wiped on the
    next attempt, and the previous (working) `extracted/` is only
    replaced on full success.

    Raises RuntimeError with user-facing wording on any failure.
    """
    sd = sidecar_dir()
    wheel_path = os.path.join(sd, f"torch-{torch_version}+{cuda_tag}.whl")
    extracted = sidecar_extracted_dir()
    extracted_partial = extracted + ".partial"

    # 1. Disk-space pre-flight — 6 GB covers the 2.5 GB wheel + 2.5 GB
    # extraction + ~1 GB headroom for the OS and temp files.
    _check_free_space_gb(sd, needed_gb=6.0)

    # 2. Clean up any half-finished extraction from a prior killed run.
    # The atomic-rename pattern below depends on `extracted.partial`
    # being absent.
    try:
        if os.path.isdir(extracted_partial):
            shutil.rmtree(extracted_partial, ignore_errors=True)
    except Exception:
        pass

    # 3. Wipe stale `extracted/` — this is a destructive step but
    # necessary: the new wheel might rename or delete files relative
    # to the old one, and overlaying creates a Frankenstein torch.
    # Only done after the disk-space check passes so we never end up
    # with NEITHER a working old install NOR enough disk for a new
    # one.
    try:
        if os.path.isdir(extracted):
            _log(f"Removing previous installation at {extracted}")
            shutil.rmtree(extracted, ignore_errors=True)
    except Exception:
        pass

    # 4. Download.  Retries + resumes internally; raises on terminal
    # failure.
    download_wheel(url, wheel_path,
                   progress_cb=download_progress_cb,
                   cancel_cb=cancel_cb)

    if cancel_cb is not None:
        try:
            if cancel_cb():
                # Keep the wheel around — cheap resume next time.
                raise RuntimeError("Installation cancelled by user.")
        except RuntimeError:
            raise
        except Exception:
            pass

    # 5. Extract to a side directory, atomic-rename on success.
    extract_wheel(wheel_path, extracted_partial,
                  progress_cb=extract_progress_cb)

    # Verify the layout looks right before committing the rename.
    if not os.path.isfile(os.path.join(extracted_partial,
                                        "torch", "__init__.py")):
        # Don't trash the partial — keep it for forensics.
        raise RuntimeError(
            f"Extraction completed but torch/__init__.py is missing in "
            f"{extracted_partial}.  The wheel layout may be unexpected.")

    try:
        os.replace(extracted_partial, extracted)
    except OSError as exc:
        # On Windows, os.replace fails if the destination already
        # exists (we wiped it above, but Defender could've re-created
        # an empty marker).  Last-resort: shutil.move which copies if
        # rename fails.
        _log(f"  atomic rename failed ({exc}); falling back to move")
        try:
            if os.path.isdir(extracted):
                shutil.rmtree(extracted, ignore_errors=True)
            shutil.move(extracted_partial, extracted)
        except Exception as exc2:
            raise RuntimeError(
                f"Could not finalise the installation: {exc2}.  The "
                f"extracted files are at {extracted_partial}; copy them "
                f"to {extracted} manually if this keeps failing."
            ) from exc2

    # 6. Wheel no longer needed — reclaim ~2.5 GB.  Failure to delete
    # is not fatal (Defender may still be scanning it).
    try: os.remove(wheel_path)
    except Exception:
        # Try once more after a beat — Defender often holds the file
        # for 1-2 seconds on Windows.
        try:
            time.sleep(1.0)
            os.remove(wheel_path)
        except Exception:
            _log(f"  NOTE: could not delete {wheel_path} "
                 f"(probably still being scanned).  Safe to remove "
                 f"manually later.")

    if not is_installed():
        raise RuntimeError(
            "Extraction completed but torch/__init__.py is missing in "
            "the sidecar directory.  The wheel layout may be unexpected.")


# ── sys.path injection ────────────────────────────────────────────────────────
def inject_sidecar_into_sys_path() -> None:
    """Prepend the sidecar's extracted directory to sys.path so subsequent
    `import torch` resolves to the CUDA build.  No-op on non-Windows or
    when the sidecar isn't installed.  Idempotent.

    MUST be called BEFORE any `import torch` anywhere in the process.
    """
    if not is_windows():
        return
    try:
        if not is_installed():
            return
        target = sidecar_extracted_dir()
        # ABI gate: never inject a torch built for a different Python
        # version.  is_installed() already checks this, but verify again at
        # the actual injection site — a cp312 sidecar prepended to a cp313
        # process shadows the bundled torch and makes `import torch` fail
        # outright (the bug that crashed the 3.13 build).
        if not _sidecar_abi_ok(target):
            return
        # Drop any stale entry first, then put it at the very front so it
        # shadows the bundled CPU torch.
        try:
            while target in sys.path:
                sys.path.remove(target)
        except Exception:
            pass
        sys.path.insert(0, target)
    except Exception:
        # Disk / permissions errors here must NOT crash FIREFLY at startup.
        pass


# ── Uninstall ─────────────────────────────────────────────────────────────────
def uninstall() -> None:
    """Remove the sidecar directory.  Used to clean up or change CUDA
    versions."""
    try:
        sd = sidecar_dir()
        if os.path.isdir(sd):
            shutil.rmtree(sd, ignore_errors=True)
    except Exception:
        pass
