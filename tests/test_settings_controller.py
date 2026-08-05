"""Persistence boundaries for the QML settings owner."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from firefly.ui.controllers import settings_controller


class _Signal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self):
        for callback in list(self._callbacks):
            callback()


class _Application:
    def __init__(self):
        self.aboutToQuit = _Signal()


class _FakeQSettings:
    instances = []

    def __init__(self, *_args):
        self.values = {}
        self.sync_calls = 0
        type(self).instances.append(self)

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.sync_calls += 1


def test_pending_preferences_sync_at_update_quit_boundary(monkeypatch):
    """Updater-triggered app.quit must flush all buffered user preferences."""
    app = _Application()
    _FakeQSettings.instances = []
    monkeypatch.setattr(settings_controller, "QSettings", _FakeQSettings)

    class _FakeCoreApplication:
        @staticmethod
        def instance():
            return app

    # ``raising=False`` lets this regression fail against the old controller,
    # which did not consult QCoreApplication at all.
    monkeypatch.setattr(settings_controller, "QCoreApplication",
                        _FakeCoreApplication, raising=False)

    controller = settings_controller.SettingsController()
    controller.setValue("ui/font_size", "Large — 14px")
    store = _FakeQSettings.instances[0]
    assert store.sync_calls == 0

    app.aboutToQuit.emit()
    assert store.sync_calls == 1

