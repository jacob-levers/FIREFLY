"""The parallel CZI decoder must survive czifile's API rename.

czifile moved subblock decoding off the directory entry: it was
``entry.data_segment()`` up to 2024.5.22, and is ``entry.read_segment_data(czi)``
in current releases.  Every call site sits inside ``except Exception``, so on a
machine with the newer czifile the parallel decoder raised AttributeError on its
first subblock and fell back to the single-threaded read for EVERY file — one
"Parallel decode failed (...)" line in the log and otherwise invisible, at a
cost of ~3 minutes per 16k-frame recording (16.8s vs 3min measured).

These are static/shim-level checks: exercising the real decoder needs a
multi-GB compressed CZI, so the end-to-end verification is manual (see
docs/DEVELOPER notes) and this pins the contract that broke.
"""
import ast
import os

import pytest

from firefly.analysis import fa_loaders


class _NewEntry:
    """Entry exposing only the CURRENT czifile API."""
    def __init__(self, payload): self._p = payload
    def read_segment_data(self, czi):
        assert czi is not None, "the file handle must be passed through"
        return _Segment(self._p)


class _OldEntry:
    """Entry exposing only the LEGACY czifile API."""
    def __init__(self, payload): self._p = payload
    def data_segment(self):
        return _Segment(self._p)


class _Segment:
    def __init__(self, payload): self._p = payload
    def data(self, raw=False):
        assert raw is False, "decoded pixels are wanted, not the raw buffer"
        return self._p


def test_shim_handles_both_czifile_spellings():
    sentinel = object()
    assert fa_loaders._czi_subblock_data(_NewEntry(sentinel), czi="h") is sentinel
    assert fa_loaders._czi_subblock_data(_OldEntry(sentinel), czi="h") is sentinel


def test_an_entry_with_neither_api_fails_loudly():
    """A third rename must raise a named error, not be swallowed into a silent
    fallback the way the last one was."""
    class _Bare:
        pass
    with pytest.raises(AttributeError, match="read_segment_data"):
        fa_loaders._czi_subblock_data(_Bare(), czi="h")


def test_no_call_site_reaches_for_the_removed_attribute_directly():
    """All three decode sites must go through the shim."""
    src = open(fa_loaders.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "data_segment"):
            # the shim itself resolves the legacy name via getattr, not a call
            offenders.append(node.lineno)
    assert not offenders, (
        f"direct entry.data_segment() calls at lines {offenders} — route them "
        f"through _czi_subblock_data() so a czifile rename can't disable the "
        f"parallel decoder again")


def test_the_installed_czifile_satisfies_the_shim():
    """Guard against the shim going stale against the pinned dependency."""
    czifile = pytest.importorskip("czifile")
    entry_cls = getattr(czifile, "DirectoryEntryDV", None) or getattr(
        czifile, "CziDirectoryEntryDV", None)
    if entry_cls is None:
        pytest.skip("czifile does not expose its directory-entry class by name")
    assert (hasattr(entry_cls, "read_segment_data")
            or hasattr(entry_cls, "data_segment")), (
        f"czifile {czifile.__version__} exposes neither decode entry point; "
        f"_czi_subblock_data needs a new branch")
