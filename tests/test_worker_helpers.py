"""Regression tests for the worker queue / atomic-write helpers (R2-7).

These back the concurrency + data-integrity fixes (#7/#25/#28, R2-4/R2-5) that
were previously verified only by throwaway scripts — reverting any of them now
turns a committed test red.
"""
import os
import json
import queue
import time

import pytest

from firefly.firefly_worker import (
    _safe_put, _put_reliable, _atomic_write_json, _postproc_calibration,
)
from firefly.analysis.fa_constants import (
    DEFAULT_PIXEL_SIZE_UM, DEFAULT_FRAME_INTERVAL_S,
)


def test_safe_put_drops_on_full_without_blocking():
    """High-volume chatter: put_nowait, drop on a full queue, never raise."""
    q = queue.Queue(maxsize=1)
    q.put("a")
    _safe_put(q, "b")              # must not raise or block
    assert q.qsize() == 1          # "b" was dropped, "a" intact
    assert q.get_nowait() == "a"


def test_put_reliable_delivers_when_room():
    """Important/terminal messages are delivered when the queue has room."""
    q = queue.Queue(maxsize=2)
    assert _put_reliable(q, "x") is True
    assert q.get_nowait() == "x"


def test_put_reliable_is_bounded_on_full():
    """A wedged consumer must NOT let _put_reliable block forever — it gives up
    after ~timeout and returns False (the residual-deadlock fix, R2-4)."""
    q = queue.Queue(maxsize=1)
    q.put("a")
    t0 = time.monotonic()
    ok = _put_reliable(q, "b", timeout=0.3)
    dt = time.monotonic() - t0
    assert ok is False
    assert dt < 2.0                # bounded (~timeout), not indefinite


def test_atomic_write_json_success(tmp_path):
    p = str(tmp_path / "out.json")
    _atomic_write_json({"a": 1, "b": [1, 2]}, p, indent=2)
    assert json.load(open(p)) == {"a": 1, "b": [1, 2]}
    assert not os.path.exists(p + ".tmp")        # no temp left behind


def test_atomic_write_json_cleans_tmp_on_failure(tmp_path):
    """A failed os.replace (here: destination is a directory) must NOT orphan
    the .tmp file, and must re-raise so the caller's WARN fires (R2-5)."""
    d = str(tmp_path / "isdir")
    os.makedirs(d)
    with pytest.raises(Exception):
        _atomic_write_json({"a": 1}, d, indent=2)
    assert not os.path.exists(d + ".tmp")


def test_postproc_calibration_recovers_persisted_values():
    """A post-process must re-run at the ORIGINAL run's calibration, not the
    default.  params.json stores px/Δt under the canonical pixel_size_um /
    frame_interval_s keys; _postproc_calibration maps them onto the request-side
    pixel_size / frame_interval the analysis entry point actually reads.  A prior
    setdefault('frame_interval', …) keyed on the wrong name and silently used the
    default for every re-ROI of a custom-calibration run (D ∝ px²/Δt)."""
    px, fi = _postproc_calibration({"pixel_size_um": 0.065, "frame_interval_s": 0.011})
    assert px == 0.065
    assert fi == 0.011
    # And it must NOT have fallen through to the shared default.
    assert px != DEFAULT_PIXEL_SIZE_UM
    assert fi != DEFAULT_FRAME_INTERVAL_S


def test_postproc_calibration_falls_back_when_absent():
    """A missing / pre-2.67 params.json (no calibration keys) still yields the
    shared default rather than raising."""
    assert _postproc_calibration({}) == (DEFAULT_PIXEL_SIZE_UM, DEFAULT_FRAME_INTERVAL_S)
    # A zero/None stored value is treated as absent (0 px/Δt is non-physical).
    assert _postproc_calibration(
        {"pixel_size_um": 0, "frame_interval_s": None}
    ) == (DEFAULT_PIXEL_SIZE_UM, DEFAULT_FRAME_INTERVAL_S)
