"""Atomic-write helpers: an interrupted write must never leave a file at the
final path — only the cleaned-up .tmp — and a pre-existing complete file must
survive a failed re-write."""
import os

import pytest

pytest.importorskip("pandas")
import pandas as pd

from firefly.analysis.fa_io import atomic_to_csv, atomic_write


def test_atomic_to_csv_success(tmp_path):
    p = str(tmp_path / "out.csv")
    atomic_to_csv(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), p, index=False)
    assert os.path.exists(p) and not os.path.exists(p + ".tmp")
    assert pd.read_csv(p)["a"].tolist() == [1, 2]


class _BoomDF:
    """Writes some bytes to the temp path, then raises mid-write."""
    def to_csv(self, path, **kw):
        with open(path, "w") as fh:
            fh.write("partial,row\n1,2")
        raise OSError("disk full")


def test_atomic_to_csv_failure_leaves_no_partial(tmp_path):
    p = str(tmp_path / "out.csv")
    pd.DataFrame({"a": [9]}).to_csv(p, index=False)        # pre-existing complete file
    with pytest.raises(OSError):
        atomic_to_csv(_BoomDF(), p, index=False)
    # the temp file is gone and the final path still holds the OLD complete file
    assert not os.path.exists(p + ".tmp")
    assert pd.read_csv(p)["a"].tolist() == [9]


def test_atomic_write_success_and_failure(tmp_path):
    p = str(tmp_path / "out.txt")
    with atomic_write(p, "w") as fh:
        fh.write("hello")
    assert open(p).read() == "hello" and not os.path.exists(p + ".tmp")
    # a mid-write exception leaves the prior file intact, no .tmp
    with pytest.raises(RuntimeError):
        with atomic_write(p, "w") as fh:
            fh.write("partial")
            raise RuntimeError("boom")
    assert open(p).read() == "hello" and not os.path.exists(p + ".tmp")
