"""_pick_startup_theme: Dark default + one-time migration off the old QSettings
domain that reverted the theme (typically to AMOLED) on every macOS update.

The fake QSettings keeps the test off the real user plists (no pollution).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")                            # skip in the Qt-less CI image
from firefly.ui import ui_theme                           # noqa: E402

NEW = ("jacoblevers", "FIREFLY")     # primary store — persists reliably
OLD = ("FIREFLY", "sptPALM")         # old foreign domain — the bug's source


class _FakeQSettings:
    store: dict = {}

    def __init__(self, org, app): self._k = (org, app)
    def value(self, key, default=None): return type(self).store.get((self._k, key), default)
    def setValue(self, key, value): type(self).store[(self._k, key)] = value
    def sync(self): pass


def _patch(monkeypatch, values, file_theme=None):
    _FakeQSettings.store = dict(values)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    # Keep the durable plain-file store off the real user file.  By default it's
    # empty so these tests exercise the QSettings-migration fallback; pass
    # file_theme to simulate a persisted file choice.
    monkeypatch.setattr(ui_theme, "_read_theme_file", lambda: file_theme)


def test_defaults_to_dark_when_nothing_is_set(monkeypatch):
    _patch(monkeypatch, {})
    assert ui_theme._pick_startup_theme() == "Dark"


def test_reads_the_new_domain_when_present(monkeypatch):
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "Light"})
    assert ui_theme._pick_startup_theme() == "Light"


def test_qsettings_only_amoled_normalises_to_dark(monkeypatch):
    # The settings-domain collision could expose this stale value after an
    # update.  A deliberate modern AMOLED choice lives in the durable file.
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "AMOLED"})
    assert ui_theme._pick_startup_theme() == "Dark"


def test_stuck_amoled_in_old_domain_normalises_to_dark(monkeypatch):
    # The reported bug: a stale AMOLED in the old (unreliable) domain must not win.
    _patch(monkeypatch, {(OLD, "ui/app_theme"): "AMOLED"})
    assert ui_theme._pick_startup_theme() == "Dark"


def test_deliberate_light_migrates_from_old_domain(monkeypatch):
    _patch(monkeypatch, {(OLD, "ui/app_theme"): "Light"})
    assert ui_theme._pick_startup_theme() == "Light"


def test_new_domain_wins_over_old(monkeypatch):
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "Dark", (OLD, "ui/app_theme"): "AMOLED"})
    assert ui_theme._pick_startup_theme() == "Dark"


def test_unknown_value_falls_back_to_dark(monkeypatch):
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "Nonsense"})
    assert ui_theme._pick_startup_theme() == "Dark"


def test_file_store_is_preferred_over_qsettings(monkeypatch):
    # The durable plain-file store wins over QSettings — this is what survives an
    # in-app update on macOS (QSettings' domain didn't match the app bundle id).
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "Dark"}, file_theme="Light")
    assert ui_theme._pick_startup_theme() == "Light"


def test_file_store_respects_a_deliberate_amoled(monkeypatch):
    # A user who actually chose AMOLED keeps it — the fix is durability, NOT
    # forcing Dark.
    _patch(monkeypatch, {}, file_theme="AMOLED")
    assert ui_theme._pick_startup_theme() == "AMOLED"


def test_write_then_read_theme_file_round_trips(monkeypatch, tmp_path):
    # write_theme_file → _read_theme_file round-trip through a real (temp) file,
    # and _pick_startup_theme reads it back without touching QSettings.
    p = str(tmp_path / "ui_prefs.json")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: p)
    ui_theme.write_theme_file("AMOLED")
    assert ui_theme._read_theme_file() == "AMOLED"
    ui_theme.write_theme_file("Dark")
    assert ui_theme._read_theme_file() == "Dark"
    assert ui_theme._pick_startup_theme() == "Dark"


def test_default_dark_is_materialised_before_an_update_relaunch(monkeypatch, tmp_path):
    """Keeping the default Dark theme must still create the durable preference.

    The old implementation only wrote ``ui_prefs.json`` after a *change*.  A
    user who simply kept Dark therefore had no canonical preference, allowing a
    stale AMOLED value in QSettings to win again when the updated app relaunched.
    """
    from firefly.ui.controllers import theme_controller

    p = str(tmp_path / "ui_prefs.json")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: p)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "_pick_startup_theme",
                        ui_theme._pick_startup_theme)

    # First fixed-version launch: the settings-domain collision exposes stale
    # AMOLED in the *primary* domain, with no canonical file yet.  It must not
    # be mistaken for a deliberate modern AMOLED selection.
    _FakeQSettings.store = {(NEW, "ui/app_theme"): "AMOLED"}
    first = theme_controller.ThemeController()
    assert first.name == "Dark"
    assert ui_theme._read_theme_file() == "Dark"
    assert _FakeQSettings.store[(NEW, "ui/app_theme")] == "Dark"

    # Simulate the post-update settings-domain oddity that previously exposed
    # AMOLED again.  The canonical file must keep the user's Dark preference.
    _FakeQSettings.store = {(NEW, "ui/app_theme"): "AMOLED"}
    assert ui_theme._pick_startup_theme() == "Dark"


def test_clicking_active_theme_retries_persistence_without_repaint(monkeypatch,
                                                                   tmp_path):
    """A same-theme click repairs the file but is not a palette transition."""
    from firefly.ui.controllers import theme_controller

    p = str(tmp_path / "ui_prefs.json")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: p)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "_pick_startup_theme",
                        ui_theme._pick_startup_theme)
    _FakeQSettings.store = {(NEW, "ui/app_theme"): "Dark"}

    controller = theme_controller.ThemeController()
    os.remove(p)  # simulate a transiently lost/failed canonical write
    changed = []
    controller.changed.connect(lambda: changed.append(True))

    controller.setTheme("Dark")
    assert ui_theme._read_theme_file() == "Dark"
    assert changed == []


def test_no_test_may_write_the_real_user_theme_stores(monkeypatch, tmp_path):
    """A guard for the bug that actually caused "the theme keeps reverting".

    ``ThemeController`` persists on construction AND on ``setTheme``.  A test
    that builds one without redirecting both stores rewrites the developer's
    own preference — and since a live-switch test picks "the next theme after
    the current one", a user on Dark was flipped to AMOLED on every suite run.
    Assert the controller only ever touches the paths a test has redirected.
    """
    from firefly.ui.controllers import theme_controller

    real_path = ui_theme.theme_pref_path()
    before = None
    if os.path.isfile(real_path):
        with open(real_path, encoding="utf-8") as fh:
            before = fh.read()

    redirected = str(tmp_path / "ui_prefs.json")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: redirected)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    _FakeQSettings.store = {}

    c = theme_controller.ThemeController()
    c.setTheme(next(n for n in c.themes if n != c.name))

    assert os.path.isfile(redirected), "the redirected store was not written"
    after = None
    if os.path.isfile(real_path):
        with open(real_path, encoding="utf-8") as fh:
            after = fh.read()
    assert after == before, (
        "ThemeController wrote the REAL user theme file during a test")
