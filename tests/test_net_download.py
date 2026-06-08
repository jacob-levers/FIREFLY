"""Unit tests for the shared downloader's retry/backoff behaviour.

Exercises the path that bit the Windows updater: a transient server-side
HTTP 504 burst should be retried (and ridden out), while a permanent 4xx
aborts immediately.  No real network or sleeps.
"""
import urllib.error

import pytest

from firefly import net_download as nd


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
