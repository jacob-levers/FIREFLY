"""Keyboard shortcuts, their hints, and the monospace font work on every OS.

The UI was built on macOS: the Preferences shortcut used StandardKey.Preferences
(which Qt binds to nothing on Windows), the hints spelled ⌘ everywhere, and the
"Menlo" monospace fell back to a proportional font off macOS.
"""
import os
import re
import sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
_QML = os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml")


def _qml_sources():
    for dirpath, _dirs, files in os.walk(_QML):
        for f in files:
            if f.endswith(".qml"):
                path = os.path.join(dirpath, f)
                yield path, open(path, encoding="utf-8").read()


def test_preferences_shortcut_is_bound_on_every_os():
    from PySide6.QtGui import QKeySequence
    main = open(os.path.join(_QML, "Main.qml"), encoding="utf-8").read()
    assert "sequence: StandardKey.Preferences" not in main
    assert 'Shortcut { sequence: "Ctrl+,"' in main
    assert not QKeySequence("Ctrl+,").isEmpty()


def test_no_shortcut_hint_hard_codes_the_mac_command_key():
    # Comments may say ⌘; strings a user sees must go through Main's modKey.
    offenders = []
    for path, src in _qml_sources():
        for n, line in enumerate(src.splitlines(), 1):
            code = line.split("//", 1)[0]
            if "⌘" in code and "modKey:" not in code:
                offenders.append(f"{os.path.basename(path)}:{n}: {line.strip()}")
    assert offenders == []


@pytest.mark.skipif(sys.platform == "darwin", reason="Menlo is native on macOS")
def test_menlo_falls_back_to_a_monospace_font():
    # Checks the registered fallbacks rather than the resolved font: the
    # offscreen platform these tests run on has no font database on Windows.
    from PySide6.QtGui import QFont
    from firefly.ui.app_qml import install_font_fallbacks
    install_font_fallbacks()
    subs = [s.lower() for s in QFont.substitutes("Menlo")]     # Qt lowercases them
    assert subs[:1] == ["consolas"]
    assert {"dejavu sans mono", "courier new"} <= set(subs)


def test_open_folder_never_builds_a_file_url_by_hand():
    pattern = re.compile(r'"file://"\s*\+')
    hits = [os.path.basename(p) for p, src in _qml_sources() if pattern.search(src)]
    assert hits == []
