"""Session-wide guards: no test opens the developer's live FIREFLY preferences
or writes FIREFLY's real data folder (theme file, saved ROIs).

The app opens every preference store with ``QSettings.defaultFormat()``
(``settings_controller.app_settings``), so pointing that format at a throwaway
INI tree isolates every store a test builds — including one a test forgot to
fake.  Without it a Mac run read and wrote the real preferences: cfprefsd
ignores $HOME, so redirecting HOME was never enough.
"""
import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_preferences(tmp_path_factory):
    try:
        from PySide6.QtCore import QSettings
    except ImportError:                     # the Qt-less core CI image
        yield
        return
    root = tmp_path_factory.mktemp("qsettings")
    previous = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root))
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.SystemScope,
                      str(root / "system"))
    yield
    QSettings.setDefaultFormat(previous)


@pytest.fixture(autouse=True, scope="session")
def _isolated_data_folder(tmp_path_factory):
    import os
    previous = os.environ.get("FIREFLY_DATA_DIR")
    os.environ["FIREFLY_DATA_DIR"] = str(tmp_path_factory.mktemp("firefly-data"))
    yield
    if previous is None:
        os.environ.pop("FIREFLY_DATA_DIR", None)
    else:
        os.environ["FIREFLY_DATA_DIR"] = previous
