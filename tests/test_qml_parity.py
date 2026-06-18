"""Phase-7 QSettings parity audit.

Locks the contract that every persisted Widgets setting (the single source of
truth ``_setting_specs``) is covered by a QML controller writing the SAME key —
so a user's configured run reproduces byte-identically after the switchover and
``firefly_worker`` never changes.  A new persisted Widgets key with no QML owner
fails this test, surfacing the gap before release.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import tempfile                                          # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

# Keys deliberately NOT persisted by the QML app (paths/mode are session-only;
# the Widgets app actively removes them on restore, see _restore_settings).
_INTENTIONAL = {"analysis/batch_folder", "analysis/mode_batch"}

# QML controllers that own settings keys beyond the sidebar schema.
_IMPORT_KEYS = {"analysis/file", "analysis/outdir", "analysis/override_px",
                "analysis/pixel_size", "analysis/override_fi", "analysis/frame_interval"}
_VISUALISE_KEYS = {"visualise/motion_colours", "visualise/cluster_point_size"}
_COMPARE_KEYS = {"compare/theme", "compare/pdf_report", "compare/outdir", "compare/stem"}
_STATS_KEYS = {"stats/alpha", "stats/correction", "stats/across_metric", "stats/strategy",
               "stats/anova3plus", "stats/nonparam", "stats/posthoc", "stats/control_group",
               "stats/dunnett", "stats/tost", "stats/tost_margin", "stats/ci_level",
               "stats/figure_stars_corrected", "stats/include_circular", "stats/circ_kappa",
               "stats/circ_rbar", "stats/circ_mu", "stats/circ_circlin"}


def _qml_covered_keys():
    from firefly.ui.controllers import sidebar_schema as S
    return ({f["key"] for f in S.FIELDS} | _IMPORT_KEYS | _VISUALISE_KEYS
            | _COMPARE_KEYS | _STATS_KEYS)


def _widgets_spec_keys():
    """The persisted-key set from the Widgets _setting_specs (single source of
    truth), read against a throwaway empty QSettings store."""
    from PySide6 import QtCore
    tmp = tempfile.mktemp(suffix=".ini")
    orig = QtCore.QSettings
    QtCore.QSettings = lambda *a, **k: orig(tmp, orig.Format.IniFormat)
    try:
        from firefly.ui.app_qt import MainWindow
        w = MainWindow()
        return {s[0] for s in w._setting_specs()}
    finally:
        QtCore.QSettings = orig


def test_every_widgets_setting_has_a_qml_owner():
    spec = _widgets_spec_keys()
    covered = _qml_covered_keys()
    gaps = spec - covered - _INTENTIONAL
    assert not gaps, f"persisted Widgets keys with no QML owner: {sorted(gaps)}"


def test_no_stale_keys_claimed():
    """Every key the QML controllers claim to own is a real persisted Widgets
    key (catches typos / renamed keys drifting out of sync)."""
    spec = _widgets_spec_keys()
    # sidebar schema + import + visualise must all be real spec keys
    from firefly.ui.controllers import sidebar_schema as S
    owned = {f["key"] for f in S.FIELDS} | _IMPORT_KEYS | _VISUALISE_KEYS
    stale = owned - spec
    assert not stale, f"QML claims keys the Widgets app doesn't persist: {sorted(stale)}"


def test_theme_namespace_shared_with_widgets():
    """ThemeController persists app_theme to the SAME store + key the Widgets
    Figures dropdown + _pick_startup_theme use, so the theme survives switchover."""
    from PySide6 import QtCore
    from firefly.ui.controllers.theme_controller import ThemeController
    # write via the controller
    t = ThemeController()
    other = next(n for n in t.themes if n != t.name)
    t.setTheme(other)
    # read back from the Widgets store/key
    s = QtCore.QSettings("FIREFLY", "sptPALM")
    assert str(s.value("ui/app_theme")) == other
    t.setTheme("Dark")            # restore


def test_sidebar_defaults_match_widgets_pristine():
    """Each sidebar-schema default equals the Widgets pristine widget value, so
    a never-configured user gets identical params from either app."""
    from firefly.ui.controllers import params_builder as pb
    # params_builder._DEFAULTS was captured from the pristine Widgets sidebar;
    # the schema defaults must agree for the keys they share.
    from firefly.ui.controllers import sidebar_schema as S
    # spot-check a representative slice (full set covered by the params parity test)
    checks = {
        "analysis/diameter": 7, "analysis/memory": 3, "analysis/search_range": 5,
        "analysis/bg_radius": 10, "analysis/max_lagtime": 20, "analysis/drift_segment": 500,
        "figures/dpi": 150, "figures/batch_dpi": 110,
    }
    for k, v in checks.items():
        assert S.BY_KEY[k]["default"] == v, k
