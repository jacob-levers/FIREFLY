"""Nudging the detection threshold by a known amount.

The threshold field was a bare text box beside a 0–50 slider.  Neither gives
fine control where it matters: a working minmass is around 0.45, so one slider
pixel is a large fraction of the value, and typing means retyping the whole
number to try 0.46.  `SpinBox` now offers opt-in stepper arrows driven by its
`step` property, and the threshold section adds a step-size selector because a
sensible increment at 0.45 and at 20 are two orders of magnitude apart.

`stepBy` carries the arithmetic (clamping, commit), and the arrows and the
Up/Down keys both go through it — so testing it covers every way to nudge.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
os.environ.setdefault("QML_DISABLE_DISK_CACHE", "1")

import pytest

pytest.importorskip("PySide6")

from test_qml_smoke import _app, qml_window                 # noqa: F401,E402

STEP_DEFAULT = 0.01


def _find(item, name):
    """Depth-first search for an item by objectName."""
    for child in item.childItems():
        if child.objectName() == name:
            return child
        found = _find(child, name)
        if found is not None:
            return found
    return None


def _step(spin, n):
    from PySide6.QtCore import QMetaObject, Qt, Q_ARG
    QMetaObject.invokeMethod(spin, "stepBy", Qt.DirectConnection, Q_ARG("QVariant", n))


@pytest.fixture
def threshold(qml_window, tmp_path):
    import numpy as np
    import tifffile

    win, qw = qml_window
    roi = qw.rootContext().contextProperty("Roi")
    stack = np.random.default_rng(2).normal(100, 2, (3, 64, 64)).astype("float32")
    stack[:, 30:33, 30:33] += 900.0
    path = tmp_path / "step.tif"
    tifffile.imwrite(path, stack, photometric="minisblack")

    win.resize(1400, 950); win.show()
    roi.editDetection(str(path))
    _app.processEvents()
    spin = _find(qw.rootObject(), "minmassSpin")
    yield roi, spin
    roi.cancel()
    win.hide()
    _app.processEvents()


def test_the_threshold_field_offers_arrows(threshold):
    """Without this the only ways to move the number are dragging a 0–50 slider
    and retyping it."""
    _roi, spin = threshold
    assert spin is not None, "the detection threshold field is not on screen"
    assert spin.property("steppers") is True


def test_an_arrow_moves_the_threshold_by_exactly_one_step(threshold):
    roi, spin = threshold
    start = roi.detectMinmass
    _step(spin, 1)
    _app.processEvents()
    assert roi.detectMinmass == pytest.approx(start + STEP_DEFAULT, abs=1e-9)
    _step(spin, -1)
    _app.processEvents()
    assert roi.detectMinmass == pytest.approx(start, abs=1e-9)


def test_stepping_down_stops_at_zero(threshold):
    """A negative threshold is not a threshold; the field must clamp rather than
    commit one."""
    roi, spin = threshold
    roi.detectMinmass = 0.005
    _app.processEvents()
    _step(spin, -1)
    _app.processEvents()
    assert roi.detectMinmass == 0.0


def test_the_step_size_is_selectable(threshold, qml_window):
    """0.01 is the right nudge at a minmass of 0.45 and useless at 20 — mass is
    in the detector's own units, so one fixed increment cannot serve both."""
    roi, spin = threshold
    _win, qw = qml_window
    picker = _find(qw.rootObject(), "minmassStepPick")
    assert picker is not None, "no way to change the increment"
    options = picker.property("options")
    options = options.toVariant() if hasattr(options, "toVariant") else options
    offered = {str(o["v"]) for o in options}
    assert {"0.001", "0.01", "0.1", "1"} <= offered

    spin.setProperty("step", 1.0)
    start = roi.detectMinmass
    _step(spin, 1)
    _app.processEvents()
    assert roi.detectMinmass == pytest.approx(start + 1.0, abs=1e-9)


def test_typing_then_moving_the_slider_no_longer_freezes_the_field(threshold):
    """SpinBox used to assign its own `value` on commit, which destroyed the
    binding to Roi.detectMinmass — so once you had typed a threshold, the field
    stopped following the slider and showed a number that was no longer in use.
    """
    from PySide6.QtCore import QMetaObject, Qt, Q_ARG
    roi, spin = threshold
    # _commit is the path typing takes; emitting `committed` would sidestep the
    # assignment under test and pass either way
    QMetaObject.invokeMethod(spin, "_commit", Qt.DirectConnection, Q_ARG("QVariant", 0.3))
    _app.processEvents()
    roi.detectMinmass = 0.8                   # as dragging the slider does
    _app.processEvents()
    assert spin.property("value") == pytest.approx(0.8, abs=1e-9)


def test_changing_the_threshold_raises_no_qml_error(threshold):
    """The commit handler called `guide.rescore()`, which was never defined.

    QML aborts the rest of a handler when one throws, so the TypeError took the
    whole arrow function with it: the mass histogram never re-scored against the
    new threshold AND the detection overlay was never re-run.  Both are the
    reason the panel exists, and neither failed loudly — the error went to the
    Qt log and the UI simply did nothing.
    """
    from PySide6.QtCore import qInstallMessageHandler

    _roi, spin = threshold
    messages = []
    previous = qInstallMessageHandler(
        lambda mode, ctx, msg: messages.append(str(msg)))
    try:
        _step(spin, 1)                      # goes through the real onCommitted
        _app.processEvents()
    finally:
        qInstallMessageHandler(previous)

    offenders = [m for m in messages
                 if "is not a function" in m or "TypeError" in m]
    assert not offenders, f"the threshold handler threw: {offenders[:3]}"
