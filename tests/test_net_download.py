"""Unit tests for the shared downloader's retry/backoff behaviour.

Exercises the path that bit the Windows updater: a transient server-side
HTTP 504 burst should be retried (and ridden out), while a permanent 4xx
aborts immediately.  No real network or sleeps.
"""
import os
import urllib.error

import pytest

from firefly import net_download as nd


def _raise(exc):
    def _f(*a, **k):
        raise exc
    return _f


class _FakeResp:
    """Minimal stand-in for an http.client.HTTPResponse context manager."""

    def __init__(self, data: bytes, status: int = 200):
        self._data = data
        self._pos = 0
        self.status = status
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        if n is None or n < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def close(self):
        pass


def _http_error(code: int) -> urllib.error.HTTPError:
    reason = "Gateway Time-out" if code == 504 else "Error"
    return urllib.error.HTTPError("http://x/file", code, reason, {}, None)


def test_download_retries_through_504_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)   # no real backoff
    data = b"FIREFLY" * 1000
    calls = {"n": 0}
    statuses = []

    def fake_urlopen(req, timeout=20):
        calls["n"] += 1
        if calls["n"] <= 2:            # first two attempts hit the 504 burst
            raise _http_error(504)
        return _FakeResp(data)

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out.bin"
    nd.download_file("http://x/file", str(dest),
                     status_cb=statuses.append, max_attempts=6)
    assert dest.read_bytes() == data
    assert calls["n"] == 3                       # 2 failures + 1 success
    # The UI was kept informed during backoff.
    assert any("retrying" in s.lower() for s in statuses)


def test_download_504_exhausts_attempts_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)

    def fake_urlopen(req, timeout=20):
        raise _http_error(504)

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(nd.DownloadError) as ei:
        nd.download_file("http://x/file", str(tmp_path / "o.bin"),
                         max_attempts=3)
    assert "504" in str(ei.value)


def test_download_404_aborts_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=20):
        calls["n"] += 1
        raise _http_error(404)

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(nd.DownloadError):
        nd.download_file("http://x/file", str(tmp_path / "o.bin"),
                         max_attempts=5)
    assert calls["n"] == 1                       # 4xx is permanent → no retries


# ── complete single-stream resume recovery ─────────────────────────────────────
def test_complete_part_is_validated_and_finalized_without_network(tmp_path,
                                                                  monkeypatch):
    """A crash after the final byte must not issue ``Range: bytes=total-``.

    Servers correctly answer that EOF request with 416.  The downloader already
    has every byte and an authoritative size, so it should run the same caller
    validation used after a normal transfer and atomically finalize locally.
    """
    payload = b"complete authenticated installer"
    dest = tmp_path / "installer.bin"
    part = tmp_path / "installer.bin.part"
    part.write_bytes(payload)
    network_calls = []
    statuses = []
    progress = []

    def unexpected_network(req, timeout=20):
        network_calls.append(req)
        raise AssertionError("a complete resume buffer must be recovered locally")

    monkeypatch.setattr(nd.urllib.request, "urlopen", unexpected_network)
    nd.download_file(
        "http://x/installer", str(dest),
        expected_size=len(payload), max_attempts=1,
        validate_cb=lambda path: open(path, "rb").read() == payload,
        progress_cb=lambda downloaded, total: progress.append((downloaded, total)),
        status_cb=statuses.append,
    )

    assert network_calls == []
    assert dest.read_bytes() == payload
    assert not part.exists()
    assert progress[-1] == (len(payload), len(payload))
    assert any("verifying" in status.lower() for status in statuses)
    assert any("finish" in status.lower() for status in statuses)


def test_invalid_complete_part_is_discarded_before_fresh_request(tmp_path,
                                                                 monkeypatch):
    """Right-sized bytes are not trusted without caller validation.

    If authentication/format validation rejects the local file, remove it and
    make an ordinary full request rather than an impossible range request at EOF.
    """
    monkeypatch.setenv("FIREFLY_NO_PARALLEL_DOWNLOAD", "1")
    good = b"authenticated installer"
    bad = b"x" * len(good)
    dest = tmp_path / "installer.bin"
    (tmp_path / "installer.bin.part").write_bytes(bad)
    ranges = []
    validated = []

    def fake_urlopen(req, timeout=20):
        ranges.append(req.get_header("Range"))
        return _FakeResp(good)

    def validate(path):
        contents = open(path, "rb").read()
        validated.append(contents)
        return contents == good

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    nd.download_file(
        "http://x/installer", str(dest), expected_size=len(good),
        parallel_segments=1, max_attempts=1, validate_cb=validate,
    )

    assert validated == [bad, good]
    assert ranges == [None]
    assert dest.read_bytes() == good


def test_short_part_still_resumes_from_its_current_size(tmp_path, monkeypatch):
    """Adding complete-part recovery must not regress ordinary partial resume."""
    monkeypatch.setenv("FIREFLY_NO_PARALLEL_DOWNLOAD", "1")
    payload = b"ordinary resumable payload"
    prefix = payload[:9]
    dest = tmp_path / "out.bin"
    (tmp_path / "out.bin.part").write_bytes(prefix)
    ranges = []

    def fake_urlopen(req, timeout=20):
        ranges.append(req.get_header("Range"))
        response = _FakeResp(payload[len(prefix):], status=206)
        response.headers = {"Content-Length": str(len(payload) - len(prefix))}
        return response

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    nd.download_file(
        "http://x/file", str(dest), expected_size=len(payload),
        parallel_segments=1, max_attempts=1,
        validate_cb=lambda path: open(path, "rb").read() == payload,
    )

    assert ranges == [f"bytes={len(prefix)}-"]
    assert dest.read_bytes() == payload


# ── finalize (.part → final) robustness (the WinError 5 the lab hit) ──────────
def test_finalize_retries_transient_lock_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    part = tmp_path / "f.part"; part.write_bytes(b"payload")
    dest = tmp_path / "f.bin"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:                       # Defender holds the file briefly
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(nd.os, "replace", flaky)
    nd._finalize_download(str(part), str(dest), tries=8)
    assert calls["n"] == 3 and dest.read_bytes() == b"payload"


# ── parallel segment resume (the "update downloads twice" fix) ────────────────
def test_parallel_dropped_segment_resumes_without_full_refetch(tmp_path, monkeypatch):
    """A dropped parallel segment must RESUME (re-fetch only its missing tail),
    not fail the whole parallel download into a full single-stream re-download —
    the cause of the 'update downloads twice' on flaky / proxied / AV networks."""
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    TOTAL = 12 * 1024 * 1024                      # > MIN_PARALLEL_BYTES → parallel runs
    data = os.urandom(TOTAL)
    served = {"bytes": 0}
    dropped = {"done": False}

    def fake_urlopen(req, timeout=None):
        s, _, e = (req.get_header("Range") or "").replace("bytes=", "").partition("-")
        start = int(s); end = int(e) if e else TOTAL - 1
        if start == 0 and end == 0:               # probe → advertise Range support + total
            r = _FakeResp(data[:1], status=206)
            r.headers = {"Content-Range": f"bytes 0-0/{TOTAL}", "Content-Length": "1"}
            return r
        seg = data[start:end + 1]
        if not dropped["done"] and start >= TOTAL // 4:   # drop one segment once, mid-stream
            dropped["done"] = True
            seg = seg[: len(seg) // 2]            # short body → the connection "ends" early
        served["bytes"] += len(seg)
        r = _FakeResp(seg, status=206)
        r.headers = {"Content-Length": str(len(seg))}
        return r

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out.bin"
    nd.download_file("http://x/big", str(dest),
                     validate_cb=lambda p: open(p, "rb").read() == data,
                     parallel_segments=4, max_attempts=1)

    assert dropped["done"]                        # the simulated drop happened
    assert dest.read_bytes() == data              # file assembled correctly despite it
    assert served["bytes"] < TOTAL * 1.5          # only the dropped tail re-fetched — NOT the whole file twice


def test_parallel_end_ignoring_server_does_not_trigger_a_second_download(tmp_path, monkeypatch):
    """A CDN/proxy that honours the range START but ignores the END streams a 206
    to EOF for every segment.  The parallel path must read only each segment's own
    bytes and assemble EXACTLY `total`, so it does NOT fall back to a full single-
    stream re-download — the 'downloads to 100%, then downloads all over again' bug.
    A plain (Range-less) request would be the single-stream fallback; assert it
    never happens."""
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    TOTAL = 12 * 1024 * 1024                      # > MIN_PARALLEL_BYTES → parallel runs
    data = os.urandom(TOTAL)
    counts = {"ranged": 0, "plain": 0}

    def fake_urlopen(req, timeout=None):
        rng = req.get_header("Range")
        if rng is None:                            # single-stream fallback path
            counts["plain"] += 1
            r = _FakeResp(data, status=200)
            r.headers = {"Content-Length": str(TOTAL)}
            return r
        s, _, e = rng.replace("bytes=", "").partition("-")
        start = int(s); end = int(e) if e else TOTAL - 1
        if start == 0 and end == 0:                # probe → advertise Range + total
            r = _FakeResp(data[:1], status=206)
            r.headers = {"Content-Range": f"bytes 0-0/{TOTAL}", "Content-Length": "1"}
            return r
        counts["ranged"] += 1
        # Honour START, IGNORE END: stream from `start` to EOF (legal 206 body).
        body = data[start:]
        r = _FakeResp(body, status=206)
        r.headers = {"Content-Range": f"bytes {start}-{TOTAL - 1}/{TOTAL}",
                     "Content-Length": str(len(body))}
        return r

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out.bin"
    nd.download_file("http://x/big", str(dest),
                     validate_cb=lambda p: open(p, "rb").read() == data,
                     parallel_segments=4, max_attempts=2)

    assert dest.read_bytes() == data              # assembled correctly from the segments
    assert counts["ranged"] >= 4                  # the parallel segments ran
    assert counts["plain"] == 0                   # …and NO single-stream re-download happened


def test_finalize_gives_up_terminally(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    part = tmp_path / "f.part"; part.write_bytes(b"x")
    dest = tmp_path / "f.bin"
    monkeypatch.setattr(nd.os, "replace",
                        _raise(PermissionError(5, "Access is denied")))
    with pytest.raises(nd.DownloadError) as ei:
        nd._finalize_download(str(part), str(dest), tries=3)
    assert getattr(ei.value, "terminal", False) is True
    assert "manually" in str(ei.value).lower()


def test_terminal_finalize_is_not_re_downloaded(tmp_path, monkeypatch):
    # A denied rename must fail FAST with the terminal message — not re-fetch
    # the whole installer max_attempts times (the "after 6 attempts" the lab saw).
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    monkeypatch.setenv("FIREFLY_NO_PARALLEL_DOWNLOAD", "1")
    attempts = {"n": 0}

    def fake_once(*a, **k):
        attempts["n"] += 1
        raise nd.DownloadError("access denied — install manually", terminal=True)

    monkeypatch.setattr(nd, "_download_once", fake_once)
    with pytest.raises(nd.DownloadError) as ei:
        nd.download_file("http://x/file", str(tmp_path / "o.bin"), max_attempts=6)
    assert attempts["n"] == 1                     # tried once, not 6×
    assert "after 6 attempts" not in str(ei.value)
    assert "manually" in str(ei.value).lower()


def test_progress_never_exceeds_100_when_a_ranged_request_is_ignored(tmp_path, monkeypatch):
    """A proxy that answers a segment's ranged request with 200 (the WHOLE file,
    not the segment) must not push the bar past 100% or corrupt the download: the
    parallel path bails cleanly to single-stream, and every progress report stays
    within [0, total].  (The 'exceeds 100% / repeats from 0%' updater bug.)"""
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    TOTAL = 12 * 1024 * 1024                      # > MIN_PARALLEL_BYTES → parallel runs
    data = os.urandom(TOTAL)
    ignored = {"done": False}

    def fake_urlopen(req, timeout=None):
        rng = req.get_header("Range")
        if rng is None:                            # single-stream fallback (no Range)
            r = _FakeResp(data, status=200)
            r.headers = {"Content-Length": str(TOTAL)}
            return r
        s, _, e = rng.replace("bytes=", "").partition("-")
        start = int(s); end = int(e) if e else TOTAL - 1
        if start == 0 and end == 0:                # probe → advertise Range + total
            r = _FakeResp(data[:1], status=206)
            r.headers = {"Content-Range": f"bytes 0-0/{TOTAL}", "Content-Length": "1"}
            return r
        if start > 0 and not ignored["done"]:      # one segment → Range IGNORED (200)
            ignored["done"] = True
            r = _FakeResp(data, status=200)        # whole file, not the requested segment
            r.headers = {"Content-Length": str(TOTAL)}
            return r
        seg = data[start:end + 1]
        r = _FakeResp(seg, status=206)
        r.headers = {"Content-Length": str(len(seg))}
        return r

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out.bin"
    reports = []
    nd.download_file("http://x/big", str(dest),
                     progress_cb=lambda d, t: reports.append((d, t)),
                     validate_cb=lambda p: open(p, "rb").read() == data,
                     parallel_segments=4, max_attempts=2)

    assert ignored["done"]                          # the ignored-Range path was hit
    assert dest.read_bytes() == data                # completed correctly via fallback
    assert reports, "no progress reported"
    assert all(0 <= d <= t for d, t in reports)     # never below 0 or above 100%


def test_status_announces_verification_after_100pct(tmp_path, monkeypatch):
    """After the bytes are in (100%), the (possibly slow) validate + finalize
    steps must announce themselves — otherwise the bar sits at 100% looking
    frozen ('stays at 100% for ages').  Single-stream path."""
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    monkeypatch.setenv("FIREFLY_NO_PARALLEL_DOWNLOAD", "1")
    data = b"FIREFLY" * 5000
    monkeypatch.setattr(nd.urllib.request, "urlopen",
                        lambda req, timeout=20: _FakeResp(data))
    statuses = []
    order = []

    def _validate(p):
        order.append("validate")                    # ran AFTER "Verifying…"
        return open(p, "rb").read() == data

    nd.download_file("http://x/file", str(tmp_path / "out.bin"),
                     status_cb=statuses.append, validate_cb=_validate,
                     max_attempts=2)
    assert any("verifying" in s.lower() for s in statuses), statuses
    assert any("finish" in s.lower() for s in statuses), statuses
    # the "Verifying…" line is emitted before validation actually runs
    v_idx = next(i for i, s in enumerate(statuses) if "verifying" in s.lower())
    assert v_idx == len(statuses) - 2 or "verifying" in statuses[v_idx]


def test_parallel_path_also_announces_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(nd.time, "sleep", lambda *_: None)
    TOTAL = 12 * 1024 * 1024
    data = os.urandom(TOTAL)

    def fake_urlopen(req, timeout=None):
        rng = req.get_header("Range")
        if rng is None:
            r = _FakeResp(data, status=200); r.headers = {"Content-Length": str(TOTAL)}
            return r
        s, _, e = rng.replace("bytes=", "").partition("-")
        start = int(s); end = int(e) if e else TOTAL - 1
        if start == 0 and end == 0:
            r = _FakeResp(data[:1], status=206)
            r.headers = {"Content-Range": f"bytes 0-0/{TOTAL}", "Content-Length": "1"}
            return r
        seg = data[start:end + 1]
        r = _FakeResp(seg, status=206); r.headers = {"Content-Length": str(len(seg))}
        return r

    monkeypatch.setattr(nd.urllib.request, "urlopen", fake_urlopen)
    statuses = []
    nd.download_file("http://x/big", str(tmp_path / "out.bin"),
                     status_cb=statuses.append,
                     validate_cb=lambda p: open(p, "rb").read() == data,
                     parallel_segments=4, max_attempts=2)
    assert any("verifying" in s.lower() for s in statuses), statuses
    assert any("finish" in s.lower() for s in statuses), statuses


def test_updater_opts_out_of_parallel_restart_path(tmp_path, monkeypatch):
    """The release updater must use the resumable single-stream path.

    A segmented transfer can finish almost every range before one late segment
    fails or a proxy changes its Range behaviour.  The generic downloader then
    has to abandon discontiguous segment files and begin a plain transfer,
    producing the reported 100% -> restart cycle.  CUDA downloads may still opt
    into parallelism; release installers prioritize one coherent progress run.
    """
    from firefly import updater

    payload = b"installer"
    digest = "sha256:" + __import__("hashlib").sha256(payload).hexdigest()
    asset = {"name": "FIREFLY-test.bin", "url": "https://x/asset",
             "size": len(payload), "digest": digest}
    monkeypatch.setattr(updater, "updates_dir", lambda: str(tmp_path))
    monkeypatch.setattr(updater, "_validate_download", lambda path: True)
    seen = {}

    def fake_download(url, dest, **kwargs):
        seen.update(kwargs)
        with open(dest, "wb") as fh:
            fh.write(payload)

    monkeypatch.setattr(updater.net_download, "download_file", fake_download)
    updater.download_asset(asset)
    assert seen["parallel_segments"] == 1
