"""Lossless startup-theme selection and durable preference migration.

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


def test_primary_qsettings_amoled_is_preserved(monkeypatch):
    # This can be a legitimate pre-durable-store preference.  The reported
    # resets came from a live-switch smoke test writing the real store, so the
    # value itself is not evidence that it is stale.
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "AMOLED"})
    assert ui_theme._pick_startup_theme() == "AMOLED"


def test_foreign_domain_amoled_is_preserved_when_it_is_the_only_choice(monkeypatch):
    # There is no provenance marker that can distinguish an old deliberate
    # AMOLED selection from an artefact, so migration must be lossless.
    _patch(monkeypatch, {(OLD, "ui/app_theme"): "AMOLED"})
    assert ui_theme._pick_startup_theme() == "AMOLED"


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
    assert ui_theme.write_theme_file("AMOLED") is True
    assert ui_theme._read_theme_file() == "AMOLED"
    assert ui_theme.write_theme_file("Dark") is True
    assert ui_theme._read_theme_file() == "Dark"
    assert ui_theme._pick_startup_theme() == "Dark"


def test_primary_amoled_is_losslessly_materialised_before_update(monkeypatch, tmp_path):
    """A legitimate legacy AMOLED choice becomes a durable preference.

    The live QML test, rather than AMOLED itself, caused the reported reset.
    Reclassifying all QSettings-only AMOLED choices as stale would lose real
    preferences during the migration intended to protect them.
    """
    from firefly.ui.controllers import theme_controller

    p = str(tmp_path / "ui_prefs.json")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: p)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)

    # First fixed-version launch: preserve and materialise the primary value.
    _FakeQSettings.store = {(NEW, "ui/app_theme"): "AMOLED"}
    first = theme_controller.ThemeController()
    assert first.name == "AMOLED"
    assert ui_theme._read_theme_file() == "AMOLED"
    assert _FakeQSettings.store[(NEW, "ui/app_theme")] == "AMOLED"

    # The durable file wins even if a later settings value disagrees.
    _FakeQSettings.store = {(NEW, "ui/app_theme"): "Dark"}
    assert ui_theme._pick_startup_theme() == "AMOLED"


def test_corrupt_durable_file_is_never_replaced_by_startup_fallback(monkeypatch,
                                                                    tmp_path):
    """A read failure is not the same thing as an absent preference.

    Startup may use a safe in-memory fallback, but must leave both the unreadable
    durable file and the compatibility store untouched for recovery.
    """
    from firefly.ui.controllers import theme_controller

    p = tmp_path / "ui_prefs.json"
    original = "{ definitely-not-valid-json"
    p.write_text(original, encoding="utf-8")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: str(p))
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    _FakeQSettings.store = {}

    controller = theme_controller.ThemeController()

    assert controller.name == "Dark"
    assert p.read_text(encoding="utf-8") == original
    assert (NEW, "ui/app_theme") not in _FakeQSettings.store


def test_permission_denied_is_distinct_from_a_missing_theme_file(monkeypatch):
    def denied(*_args, **_kwargs):
        raise PermissionError("synthetic denial")

    monkeypatch.setattr("builtins.open", denied)
    assert ui_theme._read_theme_file() is ui_theme._THEME_FILE_UNREADABLE


def test_missing_theme_is_safely_materialised_as_dark(monkeypatch, tmp_path):
    from firefly.ui.controllers import theme_controller

    p = tmp_path / "ui_prefs.json"
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: str(p))
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    _FakeQSettings.store = {}

    controller = theme_controller.ThemeController()

    assert controller.name == "Dark"
    assert ui_theme._read_theme_file() == "Dark"
    assert _FakeQSettings.store[(NEW, "ui/app_theme")] == "Dark"


def test_qsettings_read_status_error_never_materialises_dark(monkeypatch,
                                                             tmp_path):
    """Qt reports many settings failures through status(), not exceptions."""
    from firefly.ui.controllers import theme_controller

    class _ReadErrorQSettings(_FakeQSettings):
        writes = []

        def status(self):
            return "synthetic-access-error"

        def setValue(self, key, value):
            type(self).writes.append((self._k, key, value))

    p = tmp_path / "ui_prefs.json"
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: str(p))
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _ReadErrorQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _ReadErrorQSettings)
    _ReadErrorQSettings.store = {}
    _ReadErrorQSettings.writes = []

    controller = theme_controller.ThemeController()

    assert controller.name == "Dark"  # safe in-memory fallback
    assert not p.exists()
    assert not any(key == "ui/app_theme"
                   for _domain, key, _value in _ReadErrorQSettings.writes)


@pytest.mark.parametrize("name", ["Dark", "AMOLED", "Light"])
def test_valid_durable_theme_is_not_rewritten_during_startup(monkeypatch,
                                                              tmp_path, name):
    """All explicit durable choices remain authoritative without a write-back."""
    from firefly.ui.controllers import theme_controller

    p = tmp_path / "ui_prefs.json"
    p.write_text(f'{{"app_theme":"{name}","future_key":7}}', encoding="utf-8")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: str(p))
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    _FakeQSettings.store = {}
    before = p.read_bytes()

    controller = theme_controller.ThemeController()

    assert controller.name == name
    assert p.read_bytes() == before
    assert (NEW, "ui/app_theme") not in _FakeQSettings.store


def test_explicit_theme_change_reports_persistence_failure(monkeypatch, tmp_path):
    from firefly.ui.controllers import theme_controller

    p = tmp_path / "ui_prefs.json"
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: str(p))
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
    _FakeQSettings.store = {}
    controller = theme_controller.ThemeController()

    monkeypatch.setattr(ui_theme, "write_theme_file", lambda _name: False)
    assert controller.setTheme("Light") is False
    assert controller.name == "Light"  # session switch still succeeds


def test_relative_xdg_config_home_is_ignored(monkeypatch, tmp_path):
    """The XDG spec requires XDG_CONFIG_HOME to be an absolute path."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative-config")
    home = str(tmp_path / "home")
    assert ui_theme._xdg_config_home(home) == os.path.join(home, ".config")


def test_clicking_active_theme_retries_persistence_without_repaint(monkeypatch,
                                                                   tmp_path):
    """A same-theme click repairs the file but is not a palette transition."""
    from firefly.ui.controllers import theme_controller

    p = str(tmp_path / "ui_prefs.json")
    monkeypatch.setattr(ui_theme, "theme_pref_path", lambda: p)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)
    monkeypatch.setattr(theme_controller, "QSettings", _FakeQSettings)
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
