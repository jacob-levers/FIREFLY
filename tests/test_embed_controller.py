"""EmbedController visibility: the native Visualise viewer island must only show
on the Visualise tab — and must not 'escape' onto another tab when a modal
(Preferences) is closed while you're elsewhere."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeIsland:
    """Stand-in for a native island widget — records show/hide; every other
    QWidget call (setGeometry / raise_ / setMask / reset_view …) is a no-op."""
    def __init__(self):
        self.visible = None        # None → never touched

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def __getattr__(self, _name):
        return lambda *a, **k: None


def test_viewer_does_not_escape_after_modal_on_another_tab():
    from firefly.ui.controllers.embed_controller import EmbedController, VISUALISE_TAB
    ec = EmbedController()
    viewer = _FakeIsland()
    ec.setIslands(viewer=viewer)
    ec.setViewerContent(True)                      # a run/tracks is loaded

    # on the Visualise tab → the viewer island shows
    ec.onLocationChanged(VISUALISE_TAB, "main")
    _app.processEvents()
    assert viewer.visible is True

    # switch to the Import tab → it hides
    ec.onLocationChanged(0, "main")
    assert viewer.visible is False

    # open Preferences (modal) and close it WHILE STILL ON IMPORT
    ec.setModalOpen(True)
    assert viewer.visible is False
    ec.setModalOpen(False)
    assert viewer.visible is False                 # must NOT reappear over Import

    # back on Visualise → shows again
    ec.onLocationChanged(VISUALISE_TAB, "main")
    assert viewer.visible is True


def test_modal_restores_viewer_when_still_on_visualise():
    from firefly.ui.controllers.embed_controller import EmbedController, VISUALISE_TAB
    ec = EmbedController()
    viewer = _FakeIsland()
    ec.setIslands(viewer=viewer)
    ec.setViewerContent(True)
    ec.onLocationChanged(VISUALISE_TAB, "main")
    _app.processEvents()
    assert viewer.visible is True
    # open + close a modal without leaving Visualise → the viewer comes back
    ec.setModalOpen(True)
    assert viewer.visible is False
    ec.setModalOpen(False)
    assert viewer.visible is True
