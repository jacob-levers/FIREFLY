"""Phase-7 launch smoke: run_firefly.py boots each front-end to a rendered root.

Exercises the real entry path (selector → build_main_window → QML scene-graph
paint → SPTPALM_READY_MARKER), the non-frozen equivalent of the packaging smoke.
A "blank window" (e.g. a missing Qt Quick plugin once frozen, or a QML load
error) never writes the marker, so this fails. Qt-gated + offscreen; skipped in
the Qt-less CI image.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess                                        # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtQuickWidgets")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _boot(ui_env, marker, timeout=40):
    isolated_home = marker.parent / "home"
    settings_root = marker.parent / "qsettings"
    isolated_home.mkdir(exist_ok=True)
    settings_root.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["SPTPALM_READY_MARKER"] = str(marker)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_QUICK_BACKEND"] = "software"
    env["HOME"] = str(isolated_home)
    env["APPDATA"] = str(isolated_home / "AppData" / "Roaming")
    env["LOCALAPPDATA"] = str(isolated_home / "AppData" / "Local")
    env["XDG_CONFIG_HOME"] = str(isolated_home / ".config")
    if ui_env is not None:
        env["FIREFLY_UI"] = ui_env
    # Configure QSettings before importing the real entry point.  HOME/XDG
    # redirects the durable file; forcing IniFormat + an explicit path also
    # isolates macOS CFPreferences and Windows Registry-backed QSettings.
    bootstrap = (
        "import runpy,sys;"
        "from PySide6.QtCore import QSettings;"
        "f=QSettings.Format.IniFormat;"
        "QSettings.setDefaultFormat(f);"
        "QSettings.setPath(f,QSettings.Scope.UserScope,sys.argv[2]);"
        "QSettings.setPath(f,QSettings.Scope.SystemScope,sys.argv[2]);"
        "script=sys.argv[1];sys.argv=[script];"
        "runpy.run_path(script,run_name='__main__')"
    )
    proc = subprocess.Popen([sys.executable, "-c", bootstrap,
                             os.path.join(_ROOT, "run_firefly.py"),
                             str(settings_root)],
                            cwd=_ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.exists(marker):
                return True
            if proc.poll() is not None:           # exited before the marker
                return os.path.exists(marker)
            time.sleep(0.25)
        return False
    finally:
        proc.terminate()
        try:    proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_default_frontend_boots_to_ready(tmp_path):
    """The (only) QML front-end boots to a rendered root with no env override."""
    marker = tmp_path / "ready_default"
    assert _boot(None, marker), "front-end never reached the ready marker"


def test_explicit_qml_env_still_boots(tmp_path):
    """FIREFLY_UI=qml is now a no-op alias but must still boot cleanly."""
    marker = tmp_path / "ready_qml"
    assert _boot("qml", marker), "QML front-end never reached the ready marker"
