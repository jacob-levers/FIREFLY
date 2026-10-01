"""The ROI viewer with the palmTRACER-style detector selected.

That detector ignores minmass, so the threshold panel's slider would move and
nothing would change — a control that silently does nothing.  The viewer must
say so and disable it, and its live preview must use the Wavelet threshold.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

PT_LABEL = "Wavelet — palmTRACER-style (CPU)"


class _Settings(dict):
    def get_str(self, k, d=""): return dict.get(self, k, d)
    def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
    def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v
    def sync(self): pass


@pytest.fixture
def movie(tmp_path):
    import tifffile
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:48, 0:48]
    frames = []
    for _ in range(3):
        img = np.full((48, 48), 1000.0) + rng.normal(0, 15, (48, 48))
        for cy, cx, a in ((14, 15, 2000), (30, 33, 1500)):
            img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.2 ** 2))
        frames.append(img)
    path = tmp_path / "m.tif"
    tifffile.imwrite(path, np.stack(frames).astype(np.uint16), photometric="minisblack")
    return str(path)


def _roi(settings):
    from firefly.ui.controllers.roi_controller import RoiController
    from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore
    return RoiController(store=RoiStore(), settings=settings, override_store=RoiOverrideStore())


def test_minmass_applies_to_the_usual_detectors():
    assert _roi(_Settings({"analysis/backend": "Auto"})).minmassApplies is True


def test_minmass_does_not_apply_to_the_palmtracer_detector():
    assert _roi(_Settings({"analysis/backend": PT_LABEL})).minmassApplies is False


def test_the_live_preview_uses_the_wavelet_threshold(movie):
    from firefly.analysis.fa_localize_backends import PalmTracerWaveletBackend
    import tifffile
    frame = tifffile.imread(movie)[0].astype(np.float32)
    # a NON-default threshold that keeps the brighter spot only — the default
    # (250) keeps both, so this proves the setting reaches the preview
    count = lambda T: len(PalmTracerWaveletBackend().localise(frame[None], wavelet_threshold=T, quiet=True))
    T = next(T for T in np.arange(250.0, 1000.0, 5.0) if count(T) == 1)
    expected = count(T)
    assert len(PalmTracerWaveletBackend().localise(frame[None], wavelet_threshold=250.0, quiet=True)) == 2
    s = _Settings({"analysis/backend": PT_LABEL, "analysis/wavelet_threshold": T,
                   "analysis/diameter": 7, "analysis/bg_radius": 10})
    roi = _roi(s)
    roi.editDetection(movie)
    assert roi.spotCount == expected, roi.spotSummary
