"""The indeterminate "working" shimmer sweeps the bar's CURRENT width.

It used an XAnimator from -band.width to the bar's width, which reads both
once, when it starts.  A shimmer usually starts on a bar that isn't at its final
width yet — the update bar appears at its status text's width for a frame, and
the Process / HYPER-FLY fills start near-empty and grow — so the band looped
over that starting width forever (the left half of the update bar).  These
tests render the component and read where the band actually is.
Skipped in the Qt-less CI image, like the other Qt tests.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")
import time                                             # noqa: E402

import numpy as np                                      # noqa: E402
import pytest                                           # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtQuickWidgets")
from PySide6 import QtWidgets                           # noqa: E402
from PySide6.QtCore import (Property, QCoreApplication, QEvent, QObject,   # noqa: E402
                            QUrl, Signal)
from PySide6.QtGui import QImage                        # noqa: E402
from PySide6.QtQuickWidgets import QQuickWidget         # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

_W = 1000          # widest the bar gets (px)


class _Theme(QObject):
    changed = Signal()

    def __init__(self, reduced):
        super().__init__()
        self._reduced = reduced

    @Property(bool, notify=changed)
    def reducedMotion(self):
        return self._reduced


@pytest.fixture
def shimmer_bar(tmp_path):
    """A bar hosting IndeterminateShimmer; returns (widget, sample) where
    sample() is the lit columns of the bar in the frame rendered now."""
    from firefly.ui.app_qml import _QML_DIR
    comps = os.path.join(_QML_DIR, "components")
    qml = tmp_path / "bar.qml"
    qml.write_text(f'''import QtQuick
import "{QUrl.fromLocalFile(comps).toString()}"
Rectangle {{
    width: {_W}; height: 20; color: "black"
    property real barWidth: 100
    Rectangle {{
        width: parent.barWidth; height: 8; y: 6; clip: true; color: "#202020"
        IndeterminateShimmer {{ active: true }}
    }}
}}
''')
    made = []

    def build(reduced=False):
        w = QQuickWidget()
        theme = _Theme(reduced)
        w.rootContext().setContextProperty("Theme", theme)
        w.resize(_W, 20)
        w.setSource(QUrl.fromLocalFile(str(qml)))
        assert w.status() == QQuickWidget.Status.Ready, [e.toString() for e in w.errors()]
        w.show()
        made.append((w, theme))

        def sample():
            _app.processEvents()
            img = w.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
            row = np.frombuffer(bytes(img.constBits()), np.uint8).reshape(
                img.height(), img.bytesPerLine())[10, : img.width() * 3].reshape(-1, 3).mean(1)
            return np.where(row > 32 + 8)[0]                 # brighter than the bare track
        return w, sample

    yield build
    for w, _ in made:
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _app.processEvents()


def _watch(sample, seconds):
    reach, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        lit = sample()
        reach.append(int(lit.max()) if len(lit) else -1)
        time.sleep(0.015)
    return reach


def test_the_band_sweeps_the_whole_bar_after_the_bar_widens(shimmer_bar):
    w, sample = shimmer_bar()
    _watch(sample, 0.15)                                   # shimmer starts on a 100-px bar
    w.rootObject().setProperty("barWidth", _W)             # the bar is laid out / fills up
    reach = _watch(sample, 1.4)                            # > one 1.1-s sweep
    assert max(reach) >= 0.9 * _W, (
        f"the band only reached x = {max(reach)} on a {_W}-px bar")


def test_reduced_motion_holds_a_still_centred_band(shimmer_bar):
    w, sample = shimmer_bar(reduced=True)
    w.rootObject().setProperty("barWidth", _W)
    first = sample()
    time.sleep(0.3)
    later = sample()
    assert len(first) and np.array_equal(first, later)
    assert abs((first.min() + first.max()) / 2 - _W / 2) < 0.05 * _W
