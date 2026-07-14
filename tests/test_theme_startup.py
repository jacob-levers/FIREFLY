"""_pick_startup_theme: Dark default + one-time migration off the old QSettings
domain that reverted the theme (typically to AMOLED) on every macOS update.

The fake QSettings keeps the test off the real user plists (no pollution).
"""
from firefly.ui import ui_theme

NEW = ("jacoblevers", "FIREFLY")     # primary store — persists reliably
OLD = ("FIREFLY", "sptPALM")         # old foreign domain — the bug's source


class _FakeQSettings:
    store: dict = {}

    def __init__(self, org, app): self._k = (org, app)
    def value(self, key, default=None): return type(self).store.get((self._k, key), default)
    def setValue(self, key, value): type(self).store[(self._k, key)] = value
    def sync(self): pass


def _patch(monkeypatch, values):
    _FakeQSettings.store = dict(values)
    monkeypatch.setattr(ui_theme.QtCore, "QSettings", _FakeQSettings)


def test_defaults_to_dark_when_nothing_is_set(monkeypatch):
    _patch(monkeypatch, {})
    assert ui_theme._pick_startup_theme() == "Dark"


def test_reads_the_new_domain_when_present(monkeypatch):
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "Light"})
    assert ui_theme._pick_startup_theme() == "Light"


def test_new_domain_amoled_is_respected(monkeypatch):
    # AMOLED chosen in the reliable store still works — it stays selectable.
    _patch(monkeypatch, {(NEW, "ui/app_theme"): "AMOLED"})
    assert ui_theme._pick_startup_theme() == "AMOLED"


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
