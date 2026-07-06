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
