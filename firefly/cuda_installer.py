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
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable, List, Optional

from firefly import net_download


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
    # Forward to the shared downloader so its progress lines reach the
    # same UI log callback during a wheel download.
    net_download.set_log_callback(cb)


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


def _firefly_app_dir() -> str:
    """FIXED per-user FIREFLY config dir — %LOCALAPPDATA%\\FIREFLY on Windows,
    ~/.firefly elsewhere.  This anchor never moves, so the location pointer
    below is always discoverable even after the sidecar is relocated."""
    if is_windows():
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "FIREFLY")
    return os.path.join(os.path.expanduser("~"), ".firefly")


def _location_pointer_path() -> str:
    """JSON file recording a user-chosen sidecar base.  Lives at the fixed
    anchor so both the frozen .exe and a source run read the same value."""
    return os.path.join(_firefly_app_dir(), "cuda_location.json")


def _default_sidecar_base() -> str:
    return os.path.join(_firefly_app_dir(), "torch-cuda")


def sidecar_base() -> str:
    """The 'torch-cuda' root that holds the per-interpreter (cpXX) installs.

    Defaults to %LOCALAPPDATA%\\FIREFLY\\torch-cuda but can be relocated by the
    user (see move_install); the chosen path is persisted in a pointer file at
    the fixed anchor so it survives restarts and is honoured by every build."""
    try:
        p = _location_pointer_path()
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            base = data.get("base") if isinstance(data, dict) else None
            if base and isinstance(base, str):
                return base
    except Exception:
        pass
    return _default_sidecar_base()


def _sidecar_base_is_trusted(base: str) -> bool:
    """Whether ``base`` is safe to prepend to ``sys.path[0]`` before imports.

    The extracted sidecar dir is injected at the front of sys.path BEFORE any
    import, so "can write into this directory" literally means "can run code as
    the FIREFLY user".  Refuse the locations another principal could plant a
    malicious ``torch`` / ``sitecustomize`` into: UNC / network paths, and the
    world-writable shared system roots (ProgramData, Users\\Public, Windows).  A
    normal per-user local path (the %LOCALAPPDATA% default, or a user-chosen
    local folder) is fine.  (#22)
    """
    try:
        if not base or not isinstance(base, str):
            return False
        if base.startswith("\\\\") or base.startswith("//"):
            return False                      # UNC / network share
        ab = os.path.abspath(base)
        low = ab.replace("/", "\\").lower()
        if low.startswith("\\\\"):
            return False                      # resolved to UNC
        drive = os.path.splitdrive(ab)[0]
        if not drive.endswith(":"):
            return False                      # no local drive letter
        sysdrive = (os.environ.get("SystemDrive", "C:") + "\\").lower()
        windir = (os.environ.get("WINDIR", sysdrive + "windows")).replace(
            "/", "\\").lower()
        bad_roots = (sysdrive + "programdata", sysdrive + "users\\public", windir)
        if any(low == r or low.startswith(r + "\\") for r in bad_roots):
            return False                      # multi-user / world-writable root
        return True
    except Exception:
        return False


def set_sidecar_base(new_base: Optional[str]) -> None:
    """Persist (or clear, when None) the user-chosen sidecar base location."""
    try:
        os.makedirs(_firefly_app_dir(), exist_ok=True)
    except Exception:
        pass
    p = _location_pointer_path()
    if not new_base:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        return
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"base": str(new_base)}, fh)


def move_install(new_parent: str) -> str:
    """Relocate the whole CUDA sidecar tree under `new_parent` (a 'torch-cuda'
    folder is created inside it), update the pointer, and return the new base.

    The pointer is updated ONLY after the move succeeds, so a failed/interrupted
    move never leaves FIREFLY pointing at a half-moved install.  If nothing is
    installed yet this just records the location for future installs."""
    old_base = sidecar_base()
    new_base = os.path.abspath(os.path.join(new_parent, "torch-cuda"))
    if os.path.abspath(old_base) == new_base:
        return old_base
    if os.path.exists(new_base):
        if os.listdir(new_base):
            raise RuntimeError(
                f"Target already exists and is not empty:\n{new_base}")
        os.rmdir(new_base)
    if os.path.isdir(old_base):
        os.makedirs(os.path.dirname(new_base), exist_ok=True)
        shutil.move(old_base, new_base)   # moves every cpXX install at once
    else:
        os.makedirs(new_base, exist_ok=True)
    set_sidecar_base(new_base)
    return new_base


def sidecar_dir() -> str:
    """Per-interpreter sidecar dir, e.g. <sidecar_base>/cp313.

    Version-namespaced by interpreter ABI tag so a CUDA wheel installed under
    one Python version is never picked up by a build on another — that
    cross-version collision is what broke `import torch` after the 3.13 bump.
    Parent dirs are created on demand."""
    path = os.path.join(sidecar_base(), _current_py_tag())
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


def installed_torch_version() -> Optional[str]:
    """Version string (e.g. '2.12.0+cu130') of the torch in the sidecar, read
    from its version.py.  Works even before a restart (when the sidecar isn't
    yet importable).  None if not installed / unreadable."""
    try:
        vp = os.path.join(sidecar_extracted_dir(), "torch", "version.py")
        if os.path.isfile(vp):
            with open(vp, "r", encoding="utf-8") as fh:
                txt = fh.read()
            m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", txt)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


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


# ── Wheel discovery ───────────────────────────────────────────────────────────
# Match the app's runtime requirement (requirements.txt: torch>=2.6,<3).  The
# CUDA sidecar fully shadows the bundled CPU torch once injected, so it only
# needs to satisfy THIS constraint — it does NOT need to equal the bundled
# version.  That decoupling is what fixes the "no CUDA wheel for torch 2.12.0"
# dead-end: PyPI ships a torch newer than any Windows CUDA wheel PyTorch has
# published, so we install the newest CUDA build that's still in range instead.
_TORCH_MIN = (2, 6, 0)
_TORCH_MAX_EXCL = (3, 0, 0)

# CUDA toolkit channels to consider, NEWEST FIRST.  Probing a channel that
# doesn't exist just fails the index fetch and is skipped, so an over-broad
# list is safe and future-proof: when PyTorch adds e.g. cu131 we prepend it.
_CUDA_TAGS_NEWEST_FIRST = (
    "cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118")


def _parse_ver(s: str) -> Optional[tuple]:
    """'2.9.1' -> (2, 9, 1).  None if unparseable."""
    try:
        nums = [int(p) for p in s.split(".")[:3]]
        while len(nums) < 3:
            nums.append(0)
        return tuple(nums)
    except Exception:
        return None


def _ver_in_range(v: Optional[tuple]) -> bool:
    return v is not None and _TORCH_MIN <= v < _TORCH_MAX_EXCL


def _extract_wheel_versions(index_html: str, cuda_tag: str,
                            python_tag: str) -> List[str]:
    """From a PyTorch channel index page, return the in-range torch versions
    that have a `torch-<ver>+<cuda_tag>-<python_tag>-<python_tag>-win_amd64.whl`
    wheel, sorted newest-first.  Pure function — unit-tested without network."""
    pat = re.compile(
        r"torch-(\d+(?:\.\d+){1,2})(?:\.post\d+)?(?:%2B|\+)"
        + re.escape(cuda_tag)
        + r"-" + re.escape(python_tag) + r"-" + re.escape(python_tag)
        + r"-win_amd64\.whl")
    found = {}
    for m in pat.finditer(index_html):
        v = _parse_ver(m.group(1))
        if _ver_in_range(v):
            found[v] = m.group(1)
    return [found[v] for v in sorted(found, reverse=True)]


def _select_version(available: List[str],
                    preferred: Optional[str]) -> Optional[str]:
    """Prefer an exact (in-range) match to the bundled torch version when the
    channel actually has it; otherwise take the newest available."""
    if not available:
        return None
    if preferred:
        pv = _parse_ver(preferred)
        if _ver_in_range(pv):
            for s in available:
                if _parse_ver(s) == pv:
                    return s
    return available[0]  # newest (list is sorted desc)


def _http_get_text(url: str, timeout: float = 10.0) -> Optional[str]:
    """GET a small text/HTML resource, wrapped in the same wall-clock watchdog
    url_exists() uses so a stalled Windows TLS handshake can't wedge the
    worker thread.  Returns the decoded body, or None on any failure."""
    global _last_probe_error
    import threading
    holder = {"text": None, "done": False}

    def _do():
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "FIREFLY-CUDA-installer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                holder["text"] = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            _last_probe_error = f"{type(exc).__name__}: {exc}"
        finally:
            holder["done"] = True

    t = threading.Thread(target=_do, daemon=True, name="cuda-index-fetch")
    t.start()
    t.join(timeout=timeout + 2)
    if not holder["done"]:
        _last_probe_error = f"index fetch timed out after {timeout + 2:.0f}s"
        return None
    return holder["text"]


def discover_cuda_wheel(python_tag: Optional[str] = None,
                        preferred_version: Optional[str] = None,
                        cuda_tags: tuple = _CUDA_TAGS_NEWEST_FIRST,
                        status_cb: Optional[Callable[[str], None]] = None,
                        cancel_cb=None) -> Optional[dict]:
    """Find the best CUDA torch wheel actually PUBLISHED for this interpreter
    by reading PyTorch's per-channel indexes (newest CUDA toolkit first).

    Returns {'cuda_tag', 'version', 'url'} for the newest in-range torch in
    the first reachable channel that has one — preferring an exact match to
    `preferred_version` (the bundled torch) when that channel lists it.  None
    if nothing suitable is found (offline / proxy block / PyTorch dropped
    Windows CUDA for this Python version)."""
    if python_tag is None:
        python_tag = _current_py_tag()
    for tag in cuda_tags:
        if cancel_cb is not None and cancel_cb():
            raise RuntimeError("Installation cancelled by user.")
        if status_cb is not None:
            try: status_cb(f"Checking CUDA {tag} index…")
            except Exception: pass
        index_url = f"https://download.pytorch.org/whl/{tag}/torch/"
        _log(f"--- index {index_url} ---")
        html = _http_get_text(index_url)
        if not html:
            _log(f"  ✗ {tag}: index unreachable")
            continue
        avail = _extract_wheel_versions(html, tag, python_tag)
        if not avail:
            _log(f"  ✗ {tag}: no in-range {python_tag} win_amd64 wheel listed")
            continue
        chosen = _select_version(avail, preferred_version)
        _log(f"  ✓ {tag}: {len(avail)} candidate(s) "
             f"({avail[0]}..{avail[-1]}); chose {chosen}")
        return {"cuda_tag": tag, "version": chosen,
                "url": cuda_wheel_url(chosen, cuda_tag=tag,
                                      python_tag=python_tag)}
    return None


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
    """Download a CUDA torch wheel `url` → `dest_path` with progress +
    cancel support.

    The general-purpose download robustness (atomic write, resume, stall
    watchdog, throttled progress, retry/backoff, plus the post-download
    validity check) now lives in ``firefly.net_download``; this thin
    wrapper adds the two CUDA-specific bits: the optional Windows BITS
    path (opt-in via ``FIREFLY_USE_BITS=1``) and the wheel-flavoured
    error guidance.  BITS is opt-in only — historically it caused more
    hangs than it prevented; urllib turned out just as compatible and we
    control its timeout behaviour.
    """
    import zipfile
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
            _log(f"  BITS path unavailable ({exc!r}); falling back to urllib")
    try:
        net_download.download_file(
            url, dest_path,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            validate_cb=zipfile.is_zipfile,
            headers={"User-Agent": "FIREFLY-CUDA-installer/1.0"},
            max_attempts=max_attempts)
    except net_download.DownloadError as exc:
        # User-cancel propagates as-is; everything else gets the
        # CUDA-specific "try source install" guidance appended.
        if "cancel" in str(exc).lower():
            raise
        raise RuntimeError(
            f"CUDA wheel download failed.  {exc}\n\n"
            f"Try again later, switch networks, or install FIREFLY from "
            f"source (README → 'Enabling CUDA') to use pip's downloader "
            f"instead."
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
                             cuda_tags: tuple = _CUDA_TAGS_NEWEST_FIRST,
                             download_progress_cb=None,
                             extract_progress_cb=None,
                             cancel_cb=None,
                             status_cb: Optional[Callable[[str], None]] = None
                             ) -> str:
    """Discover the best published CUDA torch wheel for this interpreter and
    install it.  Returns the cuda_tag that was installed.

    Primary strategy: read PyTorch's per-channel wheel indexes (newest CUDA
    toolkit first) and pick the newest torch version in the app's >=2.6,<3
    range — preferring an exact match to the bundled `torch_version` when a
    channel lists it.  This decouples the sidecar from the bundled CPU torch
    version: PyPI can ship a torch newer than any Windows CUDA wheel PyTorch
    has built (the "no CUDA wheel for torch 2.12.0" case), and we still find
    the newest CUDA build that works.

    Fallback (if the index can't be read — offline / corporate proxy that
    blocks directory listings but not direct downloads): HEAD-probe the
    bundled version across the same channels, newest first.

    `status_cb(msg)` is called between attempts so the GUI can update its
    label ("Checking CUDA cu128 index…", "Found torch 2.9.1 + cu128…").
    """
    _log(f"install_cuda_torch_auto starting — bundled torch={torch_version}, "
         f"py={_current_py_tag()}, cuda_tags={cuda_tags}")

    found: Optional[dict] = None
    try:
        found = discover_cuda_wheel(
            python_tag=_current_py_tag(),
            preferred_version=torch_version,
            cuda_tags=cuda_tags,
            status_cb=status_cb,
            cancel_cb=cancel_cb)
    except RuntimeError:
        raise  # cancellation propagates
    except Exception as exc:
        _log(f"index discovery errored ({exc}); falling back to HEAD probe")

    # Fallback: legacy exact-version HEAD probe.
    if found is None:
        _log("discovery found nothing — HEAD-probing bundled version directly")
        tried_urls = []
        for tag in cuda_tags:
            if cancel_cb is not None and cancel_cb():
                raise RuntimeError("Installation cancelled by user.")
            url = cuda_wheel_url(torch_version, cuda_tag=tag)
            tried_urls.append(url)
            if status_cb is not None:
                try: status_cb(f"Checking torch {torch_version} + {tag}…")
                except Exception: pass
            if url_exists(url):
                found = {"cuda_tag": tag, "version": torch_version, "url": url}
                break

        if found is None:
            url_lines = "\n  ".join(tried_urls)
            reason = (f"\n\nLast probe error: {_last_probe_error}"
                      if _last_probe_error else "")
            raise RuntimeError(
                f"Couldn't find a CUDA build of PyTorch for this version of "
                f"FIREFLY (Python {_current_py_tag()}) at "
                f"download.pytorch.org.{reason}\n\n"
                f"FIREFLY looks for a CUDA torch in the {_TORCH_MIN[0]}.{_TORCH_MIN[1]}"
                f"–{_TORCH_MAX_EXCL[0]}.x range across recent CUDA toolkits, "
                f"and also tried the bundled version directly:\n  {url_lines}\n\n"
                f"If the error mentions a certificate/SSL or connection problem, "
                f"it's a network/proxy issue rather than a missing wheel.  "
                f"Otherwise PyTorch may not yet ship a Windows CUDA wheel for "
                f"this Python version — install FIREFLY from source and follow "
                f"the 'Enabling CUDA' section of the README to let pip resolve "
                f"a matching wheel."
            )

    chosen_tag = found["cuda_tag"]
    chosen_ver = found["version"]
    if chosen_ver != torch_version:
        _log(f"NOTE: bundled torch is {torch_version} but no CUDA wheel exists "
             f"for it; installing {chosen_ver}+{chosen_tag} instead (in-range, "
             f"shadows the bundled CPU build).")
    if status_cb is not None:
        try: status_cb(f"Found torch {chosen_ver} + {chosen_tag}, downloading…")
        except Exception: pass
    install_cuda_torch_from_url(
        found["url"], torch_version=chosen_ver, cuda_tag=chosen_tag,
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
        # Refuse to inject from an untrusted base (UNC / shared / world-writable
        # location): prepending it to sys.path[0] would let anyone who can write
        # there run code as FIREFLY.  Fall back to the bundled CPU torch.  (#22)
        if not _sidecar_base_is_trusted(sidecar_base()):
            try:
                print("  WARNING: the CUDA sidecar is in a non-trusted location "
                      "(network/shared/world-writable) — ignoring it for "
                      "safety; re-install GPU support under %LOCALAPPDATA%.")
            except Exception:
                pass
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
